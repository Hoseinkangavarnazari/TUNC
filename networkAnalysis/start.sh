#!/bin/bash

# example: ./start.sh ./pr_new.py 10

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <filename.py> <total batches>"
    exit 1
fi

FILENAME=$1
TOTAL=$2
DIRPATH=$(dirname "$(realpath "$FILENAME")")
PARENT_DIR=$(dirname "$DIRPATH")

echo "Filename: $FILENAME"
echo "Total Batches: $TOTAL"
echo "Dirpath: $DIRPATH"
echo "Parent Directory: $PARENT_DIR"

for ((i=1; i<=TOTAL; i++))
do
    cp "$FILENAME" "${DIRPATH}/temp${i}.py"
done

case "$(uname)" in
    "Linux")
        for ((i=1; i<=TOTAL; i++))
        do
            gnome-terminal -- bash -c "cd $PARENT_DIR && source ./venvDA/bin/activate && cd $DIRPATH && python3 temp${i}.py $i $TOTAL; exec bash"
        done
        ;;
    "Darwin")
        for ((i=1; i<=TOTAL; i++))
        do
            osascript -e "tell app \"Terminal\" to do script \"cd $PARENT_DIR && source ./venvDip/bin/activate && cd $DIRPATH && python3 temp${i}.py $i $TOTAL\""
        done
        ;;
esac
