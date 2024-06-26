#!/usr/bin/env python3

################################################################################
# render_template.py
# Author: Zachary Susskind (ZSusskind@utexas.edu)
#
# Renders templated (Mako) sources into SystemVerilog RTL
#
#
# MIT License
# 
# Copyright (c) 2024 The University of Texas at Austin
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
################################################################################

import os
import sys
from textwrap import dedent

from mako.template import Template

def render_template_file(template_fname, result_fname, info, config):
    template_basename = os.path.basename(template_fname)
    header_text = dedent("""\
        ////////////////////////////////////////////////////////////////////////////////
        // THIS FILE WAS AUTOMATICALLY GENERATED FROM ${filename}
        // DO NOT EDIT
        ////////////////////////////////////////////////////////////////////////////////

    """)
    header = Template(header_text).render(filename=template_basename)

    with open(template_fname, "r") as f:
        template = Template(f.read())
    rendered = template.render(info=info, config=config)
    output = header + rendered

    result_dirname = os.path.dirname(result_fname)
    os.makedirs(result_dirname, exist_ok=True)
    with open(result_fname, "w") as f:
        f.write(output)

def main():
    assert(len(sys.argv) == 3)
    render_template_file(*sys.argv[1:])

if __name__ == "__main__":
    main()

