#!/usr/bin/env python3
"""Print mapping from geom IDs to names for the door scene.

Usage:
  python scripts/print_door_geom_map.py
"""

import os
import sys

from etils import epath
import mujoco

# Ensure repo root is on path
REPO_ROOT = epath.Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from mujoco_playground._src.manipulation.leap_hand import base as leap_base  # noqa: E402


def load_door_model() -> mujoco.MjModel:
  assets = leap_base.get_assets()
  xml_path = (
      REPO_ROOT
      / "mujoco_playground"
      / "_src"
      / "manipulation"
      / "leap_hand"
      / "xmls"
      / "scene_mjx_cube.xml"
  )
  xml_str = epath.Path(xml_path).read_text()
  mj_model = mujoco.MjModel.from_xml_string(xml_str, assets=assets)
  return mj_model


def id2name(model: mujoco.MjModel, obj: mujoco.mjtObj, idx: int) -> str:
  name = mujoco.mj_id2name(model, obj, idx)
  return name if name is not None else f"<unnamed:{idx}>"


def print_geom_map(model: mujoco.MjModel) -> None:
  print("Geoms (id -> name [body], type, group, contype/conaff):")
  for gid in range(model.ngeom):
    gname = id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
    bid = int(model.geom_bodyid[gid])
    bname = id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
    gtype = int(model.geom_type[gid])
    ggroup = int(model.geom_group[gid])
    contype = int(model.geom_contype[gid])
    conaff = int(model.geom_conaffinity[gid])
    print(f"  {gid:3d}: {gname}  [body={bname}]  type={gtype}  group={ggroup}  contype={contype} conaff={conaff}")


def print_body_map(model: mujoco.MjModel) -> None:
  print("\nBodies (id -> name):")
  for bid in range(model.nbody):
    bname = id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
    print(f"  {bid:3d}: {bname}")


def print_site_map(model: mujoco.MjModel) -> None:
  print("\nSites (id -> name [body]):")
  for sid in range(model.nsite):
    sname = id2name(model, mujoco.mjtObj.mjOBJ_SITE, sid)
    bid = int(model.site_bodyid[sid])
    bname = id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
    print(f"  {sid:3d}: {sname}  [body={bname}]")


def print_joint_map(model: mujoco.MjModel) -> None:
  print("\nJoints (id -> name [body], type, range):")
  for jid in range(model.njnt):
    jname = id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
    bid = int(model.jnt_bodyid[jid])
    bname = id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
    jtype = int(model.jnt_type[jid])
    jrange = model.jnt_range[jid]
    print(f"  {jid:3d}: {jname}  [body={bname}]  type={jtype}  range=[{jrange[0]:.3f}, {jrange[1]:.3f}]")


def print_actuator_map(model: mujoco.MjModel) -> None:
  print("\nActuators (id -> name [joint], type, ctrlrange):")
  for aid in range(model.nu):
    aname = id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
    jid = int(model.actuator_trnid[aid, 0])  # actuator target joint
    jname = id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) if jid >= 0 else "none"
    atype = int(model.actuator_trntype[aid])
    ctrlrange = model.actuator_ctrlrange[aid]
    print(f"  {aid:3d}: {aname}  [joint={jname}]  type={atype}  ctrlrange=[{ctrlrange[0]:.3f}, {ctrlrange[1]:.3f}]")


def main() -> None:
  model = load_door_model()
  print_geom_map(model)
  print_body_map(model)
  print_site_map(model)
  print_joint_map(model)
  print_actuator_map(model)


if __name__ == "__main__":
  main() 