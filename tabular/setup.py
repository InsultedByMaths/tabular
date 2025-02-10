from setuptools import setup

setup(name='leach',
      author='Kianté Brantely, Tim Vieira, and Hal Daumé III',
      description='Learning to Teach',
      version='1.0',
      install_requires=[
          'arsenal',
      ],
      dependency_links=[
          'https://github.com/timvieira/arsenal.git',
      ],
      packages=['leach'],
)
