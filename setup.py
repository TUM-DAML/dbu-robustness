from setuptools import setup, find_packages

setup(name='dbu-robustness',
      version='0.1',
      description='Robustness Analysis of Dirichlet-Based Uncertainty Estimation Models',
      author='Anna-Kathrin Kopetzki, Bertrand Charpentier, Daniel Zügner',
      author_email='kopetzki@in.tum.de, charpent@in.tum.de, zugnerd@in.tum.de',
      packages=['src'],
      install_requires=['numpy', 'scipy', 'scikit-learn', 'matplotlib', 'torch', 'tqdm',
                        'sacred', 'deprecation', 'pymongo', 'pytorch-lightning>=0.9.0rc2'],
      zip_safe=False)
