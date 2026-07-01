from __future__ import annotations

import torch


def jacobian(pk_chain, q, frame_X_dict, frame_names):
    jacobian_dict = {}

    q = torch.atleast_2d(q)
    batch_size = q.shape[0]
    joint_names = pk_chain.get_joint_parameter_names()
    num_joints = len(joint_names)
    joint_name2idx = {name: idx for idx, name in enumerate(joint_names)}

    frames = [pk_chain.find_frame(name) for name in pk_chain.get_joint_parent_frame_names()]

    def idx(frame):
        return joint_name2idx[frame.joint.name]

    transfer_X = {}
    for frame in frames:
        q_frame = q[:, idx(frame)]
        if frame.joint.joint_type == "prismatic":
            q_frame = q_frame.unsqueeze(-1)
        transfer_X[idx(frame)] = frame.get_transform(q_frame).get_matrix()

    frame_X_dict = {f: frame_X_dict[f] for f in frame_X_dict if f in frame_names}

    for frame_name, frame_X in frame_X_dict.items():
        jac = torch.zeros((batch_size, 6, num_joints), dtype=pk_chain.dtype, device=pk_chain.device)

        r_wf = frame_X.get_matrix()[:, :3, :3]
        x_jf = torch.eye(4, dtype=pk_chain.dtype, device=pk_chain.device).repeat(batch_size, 1, 1)
        for frame_idx in reversed(pk_chain.parents_indices[pk_chain.frame_to_idx[frame_name]].tolist()):
            frame = pk_chain.find_frame(pk_chain.idx_to_frame[frame_idx])
            joint = frame.joint
            if joint.joint_type == "fixed":
                if joint.offset is not None:
                    x_jf = joint.offset.get_matrix() @ x_jf
                continue

            r_fj = x_jf[:, :3, :3].mT
            r_wj = r_wf @ r_fj
            p_jf_j = x_jf[:, :3, 3][:, :, None]
            w_wj_j = joint.axis[None, :, None].repeat(batch_size, 1, 1)
            if joint.joint_type == "revolute":
                jacobian_v = r_wj @ torch.cross(w_wj_j, p_jf_j, dim=1)
                jacobian_w = r_wj @ w_wj_j
            elif joint.joint_type == "prismatic":
                jacobian_v = r_wj @ w_wj_j
                jacobian_w = torch.zeros([batch_size, 3, 1], dtype=jacobian_v.dtype, device=jacobian_v.device)
            else:
                raise NotImplementedError(f"Unknown joint_type: {joint.joint_type}")

            joint_idx = joint_name2idx[joint.name]
            x_jf = transfer_X[joint_idx] @ x_jf
            jac[:, :, joint_idx] = torch.cat([jacobian_v[..., 0], jacobian_w[..., 0]], dim=1)

        jacobian_dict[frame_name] = jac
    return jacobian_dict
