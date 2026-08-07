# Installation

## Supported reference environment

- Ubuntu 24.04
- Python 3.10+ for tests and offline tooling
- Isaac Sim `6.0.1.0` with Python 3.12
- ROS 2 Jazzy
- MoveIt 2
- Techman `tm2_ros2`

Other versions may work, but the launchers intentionally reject unreviewed Isaac
package versions.

## Python tooling

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
pytest -q
```

Isaac and ROS packages are not installed into this virtual environment.

## Isaac Sim

Install Isaac Sim separately and point `ISAAC_ENV` to its environment:

```bash
export ISAAC_ENV="$HOME/isaac-work/envs/isaac-sim-6.0"
```

The launcher verifies:

- Python 3.12;
- `isaacsim==6.0.1.0`;
- `isaacsim-core==6.0.1.0`;
- `isaacsim-ros2==6.0.1.0`;
- the bundled Jazzy `rclpy` and ROS libraries.

Review NVIDIA's Omniverse EULA before setting:

```bash
export OMNI_KIT_ACCEPT_EULA=YES
```

## ROS 2 and Techman

For live modes, install ROS 2 Jazzy and build the Techman packages in a colcon
workspace. The current shell launchers look for:

```text
/opt/ros/jazzy/setup.bash
$HOME/tm2_ws_apt/install/setup.bash
```

The reference physical profile was tested with:

- TM5S description and MoveIt configuration;
- the real-hardware `watson_bringup.launch.py`;
- namespace `/watson`;
- `tm_driver` and MoveIt action endpoints under that namespace.

These site assumptions must be parameterised before using another cell.

## Network

The physical Watson profile validates an isolated robot interface and fixed
private subnet. The tracked example uses RFC 5737 documentation addresses that
cannot reach a real device. Replace every value with the reviewed cell's local
configuration; do not copy example network values blindly.

Create the ignored site profile before any physical mode:

```bash
mkdir -p local
cp config/watson-site.env.example local/watson-site.env
chmod 600 local/watson-site.env
```

Replace the example locally administered MAC address with the address observed
for the reviewed robot. The shell wrappers load this file automatically, or
you can point `TECHMAN_SITE_ENV` at another owner-controlled profile. The real
MAC is deliberately not stored in Git.

Preview mode performs no network preflight and creates no ROS graph.

## IOMMU warning

Isaac may warn that IOMMU is enabled. On a bare-metal multi-GPU Linux system,
follow NVIDIA's current guidance before relying on PCIe peer-to-peer transfers.
Do not disable IOMMU merely because the generic warning appears: virtual
machines with GPU passthrough may require it. The reference demo uses one GPU
and makes no multi-GPU/P2P validation claim.
