import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'vention_curobo'

setup(
    packages=['vention_curobo'],
    name=package_name,
    version='0.0.0',
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),  # Include launch files
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hacchu',
    maintainer_email='hacchu@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'vention_node = vention_curobo.vention_node:main',
            'ur5e_joint_state_publisher = vention_curobo.ur5e_joint_state_publisher:main',
            'pick_and_place = vention_curobo.pick_and_place:main',
            'pick_and_place_moveit = vention_curobo.pick_and_place_moveit:main',
        ],
    },
)
