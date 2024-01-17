#!/bin/bash

# example:      ./start.sh ./pr_new.py 1 5 10
#               ./start.sh ./pr_new.py 6 10 10

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <filename.py> <start batch> <end batch> <total batches>"
    exit 1
fi

FILENAME=$1
START_BATCH=$2
END_BATCH=$3
TOTAL_BATCHES=$4
DIRPATH=$(dirname "$(realpath "$FILENAME")")
PARENT_DIR=$(dirname "$DIRPATH")

# echo "Filename: $FILENAME"
# echo "Start Batch: $START_BATCH"
# echo "End Batch: $END_BATCH"
# echo "Total Batches: $TOTAL_BATCHES"
# echo "Dirpath: $DIRPATH"
# echo "Parent Directory: $PARENT_DIR"

for ((i=START_BATCH; i<=END_BATCH; i++))
do
    cp "$FILENAME" "${DIRPATH}/temp${i}.py"
done

case "$(uname)" in
    "Linux")
        for ((i=START_BATCH; i<=END_BATCH; i++))
        do
            gnome-terminal -- bash -c "cd $PARENT_DIR && source ./venvDA/bin/activate && cd $DIRPATH && python3 temp${i}.py $i $TOTAL_BATCHES; exec bash"
            echo "run commands: cd $PARENT_DIR && source ./venvDA/bin/activate && cd $DIRPATH && python3 temp${i}.py $i $TOTAL_BATCHES"
        done
        ;;
    "Darwin")
        for ((i=START_BATCH; i<=END_BATCH; i++))
        do
            osascript -e "tell app \"Terminal\" to do script \"cd $PARENT_DIR && source ./venvDip/bin/activate && cd $DIRPATH && python3 temp${i}.py $i $TOTAL_BATCHES\""
            echo "run commands: cd $PARENT_DIR && source ./venvDA/bin/activate && cd $DIRPATH && python3 temp${i}.py $i $TOTAL_BATCHES"
        done
        ;;
esac
