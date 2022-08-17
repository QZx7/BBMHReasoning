#!/usr/bin/env python

from distutils.core import setup

setup(
    name='BBMHReasoning',
    version='1.1.0.dev1',
    description="blenderbot mental health reasoning",
    packages=[
        'bbmhr',
        'bbmhr.parlai',
        'bbmhr.pipeline'
    ])
