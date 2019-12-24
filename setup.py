from setuptools import setup

with open('README.md', encoding='us-ascii') as f:
    long_description = f.read()

setup_info = dict(
    name='pytorch_model_summary',
    version='0.1.1',
    author='Alison Marczewski',
    author_email='alison.marczewski@gmail.com',
    url='https://github.com/amarczew/pytorch_model_summary',
    description='It is a Keras style model.summary() implementation for PyTorch',
    long_description_content_type='text/markdown',  # This is important!
    long_description=long_description,
    license='MIT',
    install_requires=['tqdm', 'torch', 'numpy'],
    keywords='pytorch model summary model.summary() keras',
    packages=['pytorch_model_summary'],
    python_requires='>=3.6'
)

setup(**setup_info)
