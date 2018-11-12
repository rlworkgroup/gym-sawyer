# Sawyer Docker
## Instructions
- If you are using NVIDIA graphic cards
    install nvidia-docker first.
- change directory to gym-sawyer root
    ```bash
    $ cd gym-sawyer
    ```
- build
    ```bash
    # specify gpu if you are using NVIDIA graphic cards.
    $ ./sawyer/ros/docker/build.sh [cpu | gpu ]
    ```

- customize the intera.sh script
    ```bash
    $ cd sawyer/ros/docker/internal
    $ vim intera.sh
    # Edit the 'robot_hostname' field
    # Edit the 'your_ip' field
    # Verify 'ros_version' field
    # update: ros_version='kinetic'
    ```

- run
    ```bash
    # cpu
    $ ./sawyer/ros/docker/run.sh
    # gpu
    $ ./sawyer/ros/docker/run_gpu.sh
    ```

- run launcher file
    ```bash
    $ cd gym-sawyer
    $ ./sawyer/ros/docker/run_test.sh path/to/launcher_file
    ```

## Trouble Shooting
- Notice moveit sawyer collision definition.
    ```xml
    <!-- remove controller_box in sawyer_moveit/sawyer_moveit_config/srdf/sawyer.srdf.xacro -->
    <xacro:sawyer_base tip_name="$(arg tip_name)"/>
    <!--Controller Box Collisions-->
-  <xacro:if value="$(arg controller_box)">
+  <!--xacro:if value="$(arg controller_box)">
     <xacro:include filename="$(find sawyer_moveit_config)/srdf/controller_box.srdf.xacro" />
     <xacro:controller_box/>
-  </xacro:if>
+  </xacro:if-->
   <!--Right End Effector Collisions-->
   <xacro:if value="$(arg electric_gripper)">
    ```
    ```xml
    <disable_collisions link1="head" link2="right_arm_base_link" reason="Never" />
    <disable_collisions link1="head" link2="right_l0" reason="Adjacent" />
    <disable_collisions link1="head" link2="right_l1" reason="Default" />
+   <disable_collisions link1="head" link2="right_l2" reason="Default" />
    <disable_collisions link1="head" link2="screen" reason="Adjacent" />
    <disable_collisions link1="head" link2="torso" reason="Never" />
    <disable_collisions link1="pedestal" link2="right_arm_base_link" reason="Adjacent" />
    ```
