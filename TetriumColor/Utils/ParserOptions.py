import argparse
from ..Observer import Cone
from ..ColorSpace import PolyscopeDisplayType


def AddObserverArgs(parser):

    parser.add_argument('--display_basis',
                        type=lambda choice: PolyscopeDisplayType[choice], choices=[x for x in PolyscopeDisplayType])
    parser.add_argument('--od', type=float, required=False, default=0.5, help='Optical Density')
    parser.add_argument('--dimension', type=int, required=False, default=4)
    parser.add_argument('--s_cone_peak', type=int, required=False, default=419, help='S cone peak')
    parser.add_argument('--m_cone_peak', type=int, required=False, default=530, help='M cone peak')
    parser.add_argument('--l_cone_peak', type=int, required=False, default=559, help='L cone peak')
    parser.add_argument('--q_cone_peak', type=int, required=False, default=547, help='Q cone peak')
    parser.add_argument('--macula', type=float, required=False, default=1.0, help='Macula Pigment Density')
    parser.add_argument('--lens', type=float, required=False, default=1.0, help='Lens Density')
    parser.add_argument('--template', type=str, required=False, default='neitz', choices=list(Cone.templates.keys()),
                        help='Template for the cone fundamentals used')
    parser.add_argument("--verbose", type=bool, default=True, help="Verbose output")


def AddVideoOutputArgs(parser):
    # Parser Args for Animation / Video Saving
    parser.add_argument("--output_filename", type=str, required=False,
                        help="Output file name. If none is specified, the video will not be saved.")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for the video")
    parser.add_argument("--total_frames", type=int, default=360, help="Total number of frames to render")


def AddAnimationArgs(parser):
    parser.add_argument("--position", nargs='+', type=float, default=[0, 0, 0], help="Position of the object")
    parser.add_argument("--velocity", nargs='+', type=float, default=[0, 0, 0], help="Velocity of the object")
    parser.add_argument("--rotation_speed", type=float, default=60, help="Rotation speed of the object")
    parser.add_argument("--rotation_axis", nargs='+', type=float, default=[0, 0, 1], help="Rotation axis of the object")
    parser.add_argument('--which_dir', type=str, default='q', choices=['q', 'saq', 'none'])
