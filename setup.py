from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'Package to statistically analyse seismicity.'
LONG_DESCRIPTION = 'Package for importing, processing, and standardising earthquake source parameter data;\
                    selecting mainshocks using the FET, MDET, and DDET methods;\
                    identifying foreshocks using the BP, G-IET, and ESR methods.'

setup(
        name="seispy", 
        version=VERSION,
        author="Russell Azad Khan",
        author_email="<russellazadkhan@gmail.com>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[], # add any additional packages that 
        # needs to be installed along with your package. Eg: 'caer'

        keywords=['python', 'seismology'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",,
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ]
)