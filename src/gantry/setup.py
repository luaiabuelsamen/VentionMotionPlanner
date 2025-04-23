from setuptools import find_packages, setup
import os
from glob import glob
package_name = 'gantry'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'meshes/ur5e/collision'), glob('meshes/ur5e/collision/*')),
        (os.path.join('share', package_name, 'meshes/ur5e/visual'), glob('meshes/ur5e/visual/*')),
        (os.path.join('share', package_name, 'meshes/gripper/collision'), glob('meshes/gripper/collision/*')),
        (os.path.join('share', package_name, 'meshes/gripper/visual'), glob('meshes/gripper/visual/*')),
        (os.path.join('share', package_name, 'meshes/gantry'), glob('meshes/gantry/*')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'cfg'), glob('cfg/*')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hacchu',
    maintainer_email='hacchu@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        ],
    },
)
