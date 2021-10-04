#!/usr/bin/env python

from argparse import ArgumentParser, ArgumentTypeError


def print_row_vector(template, row, compress):
    row_minus_one = str(row - 1) if isinstance(row, int) else f"{row}-1"

    print("\\begin{bmatrix}")
    print(f"{template.replace('row', '1')} \\\\")
    print(f"{template.replace('row', '2')} \\\\")
    print("\\vdots \\\\")

    if not compress:
        print(f"{template.replace('row', row_minus_one)} \\\\")

    print(f"{template.replace('row', str(row))} \\\\")
    print("\\end{bmatrix}")


def print_column_vector(template, col, compress):
    col_minus_one = str(col - 1) if isinstance(col, int) else f"{col}-1"

    print("\\begin{bmatrix}")
    print(f"{template.replace('col', '1')} &")
    print(f"{template.replace('col', '2')} &")
    print("\\cdots &")

    if not compress:
        print(f"{template.replace('col', col_minus_one)} &")

    print(f"{template.replace('col', str(col))}")
    print("\\end{bmatrix}")


def print_matrix(template, row, col, compress):

    row_minus_one = str(row - 1) if isinstance(row, int) else f"{row}-1"
    col_minus_one = str(col - 1) if isinstance(col, int) else f"{col}-1"

    print("\\begin{bmatrix}")

    # Row 1
    c1 = template.replace("col", "1").replace("row", "1")
    c2 = template.replace("col", "2").replace("row", "1")
    c3 = template.replace("col", col_minus_one).replace("row", "1")
    c4 = template.replace("col", str(col)).replace("row", "1")

    if compress:
        print(f"{c1} & {c2} & \\cdots & {c4}\\\\")
    else:
        print(f"{c1} & {c2} & \\cdots & {c3} & {c4}\\\\")

    # Row 2
    c1 = template.replace("col", "1").replace("row", "2")
    c2 = template.replace("col", "2").replace("row", "2")
    c3 = template.replace("col", col_minus_one).replace("row", "2")
    c4 = template.replace("col", str(col)).replace("row", "2")

    if compress:
        print(f"{c1} & {c2} & \\cdots & {c4}\\\\")
        print("\\vdots & \\vdots & \\ddots & \\vdots \\\\")
    else:
        print(f"{c1} & {c2} & \\cdots & {c3} & {c4}\\\\")
        print("\\vdots & \\vdots & \\ddots & \\vdots & \\vdots \\\\")

    # Row n-1
    c1 = template.replace("col", "1").replace("row", row_minus_one)
    c2 = template.replace("col", "2").replace("row", row_minus_one)
    c3 = template.replace("col", col_minus_one).replace("row", row_minus_one)
    c4 = template.replace("col", str(col)).replace("row", row_minus_one)

    if not compress:
        print(f"{c1} & {c2} & \\cdots & {c3} & {c4}\\\\")

    # Row n
    c1 = template.replace("col", "1").replace("row", str(row))
    c2 = template.replace("col", "2").replace("row", str(row))
    c3 = template.replace("col", col_minus_one).replace("row", str(row))
    c4 = template.replace("col", str(col)).replace("row", str(row))

    if compress:
        print(f"{c1} & {c2} & \\cdots & {c4}\\\\")
    else:
        print(f"{c1} & {c2} & \\cdots & {c3} & {c4}\\\\")
    print("\\end{bmatrix}")


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
        print_row_vector(template, max_row, compress)

    elif args.rowvector:
        template, max_col = args.rowvector
        print_column_vector(template, max_col, compress)

    elif args.matrix:
        template, max_row, max_col = args.matrix
        print_matrix(template, max_row, max_col, compress)

    else:
        raise ArgumentTypeError(
            "You must provide either --vertor, --rowvector, or --matrix."
        )


if __name__ == "__main__":
    main()
