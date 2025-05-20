from setuptools import setup, find_packages

with open('requirements.txt', 'r') as file:
    requirements = file.readlines()
    
setup(
   name='ib_async_trading_bot',
   version='1.0',
   license='MIT',
   description='A stock trading module',
   author='Jaskirat Singh Bhatia',
   author_email='jaskiratsinghbhatia4@gmail.com',
   packages=find_packages(),
   install_requires=requirements, #external packages as dependencies
)