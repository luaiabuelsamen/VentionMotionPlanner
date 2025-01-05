from setuptools import find_packages, setup

package_name = 'motion_plan'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/CAD', ['motion_plan/CAD/base_link.STL'])
        # ('share/' + package_name + '/CAD', ['motion_plan/CAD/Tray.stl'])
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jetson',
    maintainer_email='jetson@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'path_planner = motion_plan.path_planner:main',
            'collision = motion_plan.collision:main',
            'test_collision = motion_plan.test_collision:main',
        ],
    },
)
