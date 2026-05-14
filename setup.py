from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'weed_locator'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'srv'), glob('srv/*.srv')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'weed_fusion_node = weed_locator.weed_fusion_node:main',
            'dynamixel_node = weed_locator.dynamixel_node:main',
            'delta_gcode_bridge = weed_locator.delta_gcode_bridge:main',
            'test_ik_dynamixel = weed_locator.test_ik_dynamixel:main',
        ],
    },
)
