#!/usr/bin/env bash

html_file=~/Temp/result.html
pdf_file=
python_file=


while [ "$1" != "" ]; do
    case $1 in
        -p | --pdf )            shift
                                pdf_file=$1
                                ;;
        -h | --html )           shift
                                html_file=$1
                                ;;
        * )                     python_file=$1
                                ;;
    esac
    shift
done

echo "$python_file"
echo "$html_file"
echo "$pdf_file"

script -q ~/Temp/output.txt -c "python $python_file"

cat ~/Temp/output.txt | aha > ${html_file}
if [ "$pdf_file" != "" ]; then
   wkhtmltopdf ${html_file} ${pdf_file}
fi