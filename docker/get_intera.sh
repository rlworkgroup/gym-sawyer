#!/bin/bash

die()
{
	local _ret=$2
	test -n "$_ret" || _ret=1
	test "$_PRINT_HELP" = yes && print_help >&2
	echo "$1" >&2
	exit ${_ret}
}


begins_with_short_option()
{
	local first_option all_short_options='h'
	first_option="${1:0:1}"
	test "$all_short_options" = "${all_short_options/$first_option/}" && return 1 || return 0
}

# THE DEFAULTS INITIALIZATION - OPTIONALS
_arg_sim="off"


print_help()
{
	printf '%s\n' "The intera.sh script is a convenient script which will set up
# your ROS environment for working with intera SDK. However, it needs to be
# slightly modified to work properly, namely:
#   1. Set the ros_version as kinetic
#   2. Add the -ci arguments when the intera shell is launched in order to
#     pass commands to it in the Docker container.
# The below three modifications are only applied if "robot" argument is
# passed to the script.
#   3. Set sawyer robot's hostname
#   4. Set workstation IP address
#   5. Avoid re-setting robot_hostname. (side affect of 2nd change)
# This script takes care of downloading the intera script and applying the
# modifications described above. "
	printf 'Usage: %s [--(no-)sim] [-h|--help]\n' "$0"
	printf '\t%s\n' "-h, --help: Prints help"
}


parse_commandline()
{
	while test $# -gt 0
	do
		_key="$1"
		case "$_key" in
			--no-sim|--sim)
				_arg_sim="on"
				test "${1:0:5}" = "--no-" && _arg_sim="off"
				;;
			-h|--help)
				print_help
				exit 0
				;;
			-h*)
				print_help
				exit 0
				;;
			*)
				_PRINT_HELP=yes die "FATAL ERROR: Got an unexpected argument '$1'" 1
				;;
		esac
		shift
	done
}

parse_commandline "$@"

if [[ $_arg_sim == "on" ]]; then
    intera_path="docker/sawyer-sim/intera.sh"
else
    intera_path="docker/sawyer-robot/intera.sh"
fi

if [[ ! -f "${intera_path}" ]]; then
    intera_sh="$(printf "%s" \
        'https://raw.githubusercontent.com/RethinkRobotics/' \
        'intera_sdk/8d11d55dc4ed7ad648abb6bbe1ad75566e33ff6b/intera.sh')"
    echo "${intera_sh}"
    wget "${intera_sh}" -O "${intera_path}"
    chmod u+x "${intera_path}"

    # Set ROS kinetic
    find_text="^ros_version=\"indigo\"$"
    replacement_text="ros_version=\"kinetic\""
    sed -i -e "s/${find_text}/${replacement_text}/g" "${intera_path}"

    # Set the arguments "-ci" in intera shell
    find_text="^\${SHELL} --rcfile \${tf}$"
    replacement_text="$(printf "%s" \
        'if [[ "${2}" \&\& "${1}" == "sim" ]]; then\n'\
        '  ${SHELL} --rcfile ${tf} -ci "${2}"\n'\
        'elif [[ "${1}" \&\& "${1}" != "sim" ]]; then\n'\
        '  ${SHELL} --rcfile ${tf} -ci "${1}"\n'\
        'else\n'\
        '  ${SHELL} --rcfile ${tf}\n'\
        'fi')"
    sed -i -e "s/${find_text}/${replacement_text}/g" "${intera_path}"

    if [[ $_arg_sim == "off" ]]; then
      # Set robot hostname
      find_text="robot_hostname.local"
      replacement_text="${SAWYER_HOSTNAME}"
      sed -i -e "0,/${find_text}/s/${find_text}/${replacement_text}/g" \
        "${intera_path}"

      # Set workstation ip address
      find_text="192.168.XXX.XXX"
      replacement_text="${WORKSTATION_IP}"
      sed -i -e "0,/${find_text}/s/${find_text}/${replacement_text}/g" \
        "${intera_path}"

      #Avoid from re-setting robot_hostname variable
      find_text="robot_hostname=\"\\$"{1}"\""
      replacement_text="robot_hostname=\$robot_hostname"
      sed -i -e "s/${find_text}/${replacement_text}/g" "${intera_path}"
    fi
fi
