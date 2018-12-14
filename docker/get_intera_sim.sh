#!/usr/bin/env bash

# The intera script in sim mode is used to launch Gazebo and other ROS
# components for Sawyer. However, it needs to be slightly modified to work
# properly, namely:
#   - Set the ros_version as kinetic
#   - Add the -ci arguments when the intera shell is launched in order to
#     pass commands to it in the Docker container.
# This script takes care of downloading the intera script and applying the
# modifications described above.

intera_sim_path="docker/intera_sim.sh"

if [[ ! -f "${intera_sim_path}" ]]; then
  intera_sh="$(printf "%s" \
    'https://raw.githubusercontent.com/RethinkRobotics/' \
    'intera_sdk/8d11d55dc4ed7ad648abb6bbe1ad75566e33ff6b/intera.sh')"
  echo "${intera_sh}"
	wget "${intera_sh}" -O "${intera_sim_path}"
  chmod u+x "${intera_sim_path}"

  # Set ROS kinetic
  find_text="^ros_version=\"indigo\"$"
  replacement_text="ros_version=\"kinetic\""
  sed -i -e "s/${find_text}/${replacement_text}/g" "${intera_sim_path}"

  # Set the arguments "-ci" in intera shell
  find_text="^\${SHELL} --rcfile \${tf}$"
  replacement_text="$(printf "%s" \
    'if [[ "${2}" \&\& "${1}" == "sim" ]]; then\n'\
    '  ${SHELL} --rcfile ${tf} -ci "${2}"\n'\
    'else\n'\
    '  ${SHELL} --rcfile ${tf}\n'\
    'fi')"
  sed -i -e "s/${find_text}/${replacement_text}/g" "${intera_sim_path}"
fi
