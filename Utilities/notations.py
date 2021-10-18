#!/usr/bin/env python

from argparse import ArgumentParser, ArgumentTypeError
from typing import Union

StrOrInt = Union[str, int]


def vector(template: str, ext: StrOrInt, compress: bool, column: bool):

    sep = "\\\\" if column else "&"
    ext_minus_1 = str(ext - 1) if isinstance(ext, int) else f"{ext}-1"
    ext_minus_1_str = "" if compress else f"{template.replace('ext', ext_minus_1)}{sep}"

    return f"""
\\begin{{bmatrix}}
    {template.replace("ext", "1")} {sep}
    {template.replace("ext", "2")} {sep}
    \\vdots {sep} {ext_minus_1_str}
    {template.replace("ext", str(ext))} {sep}
\\end{{bmatrix}}"""


def column_vector(template: str, row: StrOrInt, compress: bool):
    return vector(template.replace("row", "ext"), row, compress, True)


def row_vector(template: str, col: StrOrInt, compress: bool):
    return vector(template.replace("col", "ext"), col, compress, False)


def matrix(template: str, row: StrOrInt, col: StrOrInt, compress: bool):

    row_minus_1 = str(row - 1) if isinstance(row, int) else f"{row}-1"
    col_minus_1 = str(col - 1) if isinstance(col, int) else f"{col}-1"

    row, col = str(row), str(col)

    def rstr(r, c):
        return template.replace("col", c).replace("row", r)

    def cm1(r):
        return "" if compress else (rstr(r, col_minus_1) + " &")

    extra_dots = "" if compress else "& \\vdots"
    dots_row = f"\\vdots & \\vdots & \\ddots & \\vdots {extra_dots} \\\\"

    row_minus_1_str = f"""
    {rstr(row_minus_1, "1")} &
    {rstr(row_minus_1, "2")} &
    \\cdots & {cm1(row_minus_1)}
    {rstr(row_minus_1, col)} \\\\"""

    return f"""
\\begin{{bmatrix}}
    {rstr("1", "1")} & {rstr("1", "2")} & \\cdots & {cm1("1")} {rstr("1", col)} \\\\
    {rstr("2", "1")} & {rstr("2", "2")} & \\cdots & {cm1("2")} {rstr("2", col)} \\\\
    {dots_row} {"" if compress else row_minus_1_str}
    {rstr(row, "1")} & {rstr(row, "2")} & \\cdots & {cm1(row)} {rstr(row, col)}
\\end{{bmatrix}}"""


def main():

    argparser = ArgumentParser("Print latex vectors and matrices.")

    argparser.add_argument(
        "-v",
        "--vector",
        nargs=2,
        metavar=("template", "max_row"),
        help="Print a column vector.",
    )

    argparser.add_argument(
        "-r",
        "--rowvector",
        nargs=2,
        metavar=("template", "max_col"),
        help="Print a row vector.",
    )

    argparser.add_argument(
        "-m",
        "--matrix",
        nargs=3,
        metavar=("template", "max_row", "max_col"),
        help="Print a matrix.",
    )

    argparser.add_argument(
        "--nocompress",
        action="store_true",
        help="Display the second to last row and/or column.",
    )

    args = argparser.parse_args()

    compress = not args.nocompress

    if args.vector:
        template, max_row = args.vector
        print(column_vector(template, max_row, compress))

    elif args.rowvector:
        template, max_col = args.rowvector
        print(row_vector(template, max_col, compress))

    elif args.matrix:
        template, max_row, max_col = args.matrix
        print(matrix(template, max_row, max_col, compress))

    else:
        raise ArgumentTypeError(
            "You must provide either --vertor, --rowvector, or --matrix."
        )


if __name__ == "__main__":
    main()
