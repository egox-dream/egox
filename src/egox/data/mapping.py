"""Existing modality names, HDF5 bindings, and fields selected by policy configs."""

KEY_PATH_MAP = {
    # Key Observations
    'main_image': '/observations/images/main',
    'left_wrist_image': '/observations/images/wrist_left',
    'right_wrist_image': '/observations/images/wrist_right',
    'body_pose_proprio': '/observations/proprioceptions/body_proprio',
    'eef_pose_proprio': '/observations/proprioceptions/eef_proprio',
    'left_eef_to_body_pose_proprio': '/observations/proprioceptions/left_eef_to_body_proprio',
    'right_eef_to_body_pose_proprio': '/observations/proprioceptions/right_eef_to_body_proprio',
    'left_eef_link_proprio': '/observations/proprioceptions/left_eef_link_proprio',
    'right_eef_link_proprio': '/observations/proprioceptions/right_eef_link_proprio',
    'robot_joint_pos_proprio': '/observations/proprioceptions/robot_joint_pos_proprio',
    'left_eef_joint_pos_proprio': '/observations/proprioceptions/left_eef_joint_pos_proprio',
    'right_eef_joint_pos_proprio': '/observations/proprioceptions/right_eef_joint_pos_proprio',
    # Actions
    'body_pose': '/actions/body',
    'left_eef_to_body_pose': '/actions/left_eef_to_body',
    'right_eef_to_body_pose': '/actions/right_eef_to_body',
    'left_eef_link_pose': '/actions/left_eef_link',
    'right_eef_link_pose': '/actions/right_eef_link',
    'velocity_commands': '/actions/velocity_commands',
    # Other Informations
    'eef_to_body_pose_proprio': '/observations/proprioceptions/eef_to_body_proprio',
    'eef_link_proprio': '/observations/proprioceptions/other/eef_link_proprio',
    'robot_joint_vel_proprio': '/observations/proprioceptions/robot_joint_vel_proprio',
    'eef_to_body_pose': '/actions/eef_to_body',
    'eef_link_pose': '/actions/eef_link',
}

# Availability masks, at '/masks' + the field's path; fields whose mask is missing are always available.
KEY_MASK_PATH_MAP = {
    # Key Observations
    'main_image': '/masks/observations/images/main',
    'left_wrist_image': '/masks/observations/images/wrist_left',
    'right_wrist_image': '/masks/observations/images/wrist_right',
    'body_pose_proprio': '/masks/observations/proprioceptions/body_proprio',
    'eef_pose_proprio': '/masks/observations/proprioceptions/eef_proprio',
    'left_eef_to_body_pose_proprio': '/masks/observations/proprioceptions/left_eef_to_body_proprio',
    'right_eef_to_body_pose_proprio': '/masks/observations/proprioceptions/right_eef_to_body_proprio',
    'left_eef_link_proprio': '/masks/observations/proprioceptions/left_eef_link_proprio',
    'right_eef_link_proprio': '/masks/observations/proprioceptions/right_eef_link_proprio',
    'robot_joint_pos_proprio': '/masks/observations/proprioceptions/robot_joint_pos_proprio',
    'left_eef_joint_pos_proprio': '/masks/observations/proprioceptions/left_eef_joint_pos_proprio',
    'right_eef_joint_pos_proprio': '/masks/observations/proprioceptions/right_eef_joint_pos_proprio',
    # Actions
    'body_pose': '/masks/actions/body',
    'left_eef_to_body_pose': '/masks/actions/left_eef_to_body',
    'right_eef_to_body_pose': '/masks/actions/right_eef_to_body',
    'left_eef_link_pose': '/masks/actions/left_eef_link',
    'right_eef_link_pose': '/masks/actions/right_eef_link',
    'velocity_commands': '/masks/actions/velocity_commands',
    # Other Informations
    'eef_to_body_pose_proprio': '/masks/observations/proprioceptions/eef_to_body_proprio',
    'eef_link_proprio': '/masks/observations/proprioceptions/other/eef_link_proprio',
    'robot_joint_vel_proprio': '/masks/observations/proprioceptions/robot_joint_vel_proprio',
    'eef_to_body_pose': '/masks/actions/eef_to_body',
    'eef_link_pose': '/masks/actions/eef_link',
}


def embodiment_keys(emb_dict, embodiment):
    """Observation, action and future-observation (dream) fields of an embodiment."""
    cfg = emb_dict[embodiment]
    obs_keys = [k for k in cfg['tokenizer'] if 'proprio' in k or 'image' in k]
    dream_keys = (cfg['tokenizer']['embodiment_tokenizer']['modalities']
                  if 'embodiment_dream_head' in cfg else [])
    return obs_keys, list(cfg['action_head']), dream_keys
