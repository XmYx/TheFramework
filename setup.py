from setuptools import setup, find_packages

setup(
    name='TheFramework',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    entry_points={
        'console_scripts': [
            'framework = theframework.main:main',
        ],
    },
    include_package_data=True,
    install_requires=[
        # Add your project's dependencies here
        # e.g., 'requests', 'numpy', etc.
    ],
    author='Your Name',
    author_email='your.email@example.com',
    description='A brief description of The Framework',
    # long_description=open('README.md').read(),
    # long_description_content_type='text/markdown',
    url='https://github.com/yourusername/theframework',  # Update with your project URL
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
