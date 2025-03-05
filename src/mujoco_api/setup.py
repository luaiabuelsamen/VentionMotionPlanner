from setuptools import find_packages, setup

package_name = 'mujoco_api'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    package_data={
        'mujoco': ['assets/**/*'],
    },
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jetson3',
    maintainer_email='luai.abuelsamen@mail.mcgill.ca',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mujoco_service = mujoco_api.mujoco_node:main',
        ],
    },
)
