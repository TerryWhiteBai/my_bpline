已新增通用转换脚本：

[convert_piper_lerobot_to_demos.py](</home/bw/Projects/bspline-policy/real_env/yam_teleop/convert_piper_lerobot_to_demos.py:1>)

- 读取 Piper/LeRobot 格式的 Parquet
- 按 `episode_index` 拆分 episode
- 读取 `top / left_wrist / right_wrist` 视频
- 使用 Pinocchio 对 6 个关节做 FK
- 生成末端 `arm_pos` 和 `arm_quat`
- 将 Piper 夹爪 `0~0.04 m` 归一化到 `0~1`
- 输出 `data/demos/<episode>/data.pkl + MP4`
- 默认适配 `clean_bspline_policy_unet_bspline.yaml` 的左臂 task


```bash
xacrodoc ../../piper_description/urdf/piper_with_gripper.urdf.xacro \
  > /tmp/piper_with_gripper.urdf
```

```
python convert_piper_lerobot_to_demos.py \
  --input-dir ../../piper0819 \
  --output-dir data/demos \
  --left-urdf /tmp/piper_with_gripper.urdf
  --side left \
  --left-ee-frame gripper_tcp \
  --image-width 128 \
  --image-height 128 \
  --overwrite
```

默认左臂 observation 字段包括：

```text
head_image
left_wrist_image
arm_pos_l
arm_quat_l
gripper_pos_l
```

同时保留关节状态：

```text
left_arm_joint_position
```

action 中会生成当前转换器需要的：

```text
arm_pos
arm_quat
gripper_pos
```

因此后续可以使用已有的：

```bash
python convert_to_robomimic_hdf5.py \
  --input-dir data/demos \
  --output-path ../../diffusion_policy/data/piper-left.hdf5
```

然后将训练配置中的数据路径改成：

```text
../../diffusion_policy/data/piper-left.hdf5
```

脚本已经通过语法检查和 `--help` 检查；当前环境缺少 `pyarrow/cv2/pinocchio`，所以没有在本机执行完整转换。