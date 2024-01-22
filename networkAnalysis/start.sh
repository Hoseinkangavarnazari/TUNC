#!/bin/bash

# Usage example:      ./start.sh ./pr_new.py 1 10 10 3
#                     (This will run batches from 1 to 10 in parallel groups of 3)

if [ "$#" -ne 5 ]; then
    echo "Usage: $0 <filename.py> <start batch> <end batch> <total batches> <kernel>"
    exit 1
fi

FILENAME=$1
START_BATCH=$2
END_BATCH=$3
TOTAL_BATCHES=$4
KERNEL=$5
DIRPATH=$(dirname "$(realpath "$FILENAME")")
PARENT_DIR=$(dirname "$DIRPATH")

for ((k=0; k<KERNEL; k++))
do
    COMMAND="cd $PARENT_DIR"
    case "$(uname)" in
        "Linux")
            COMMAND="${COMMAND} && source ./venvDA/bin/activate && cd $DIRPATH"
            ;;
        "Darwin")
            COMMAND="${COMMAND} && source ./venvDip/bin/activate && cd $DIRPATH"
            ;;
    esac
    
    for ((i=START_BATCH+k; i<=END_BATCH; i+=KERNEL))
    do
        cp "$FILENAME" "${DIRPATH}/temp${i}.py"
        COMMAND="${COMMAND} && python3 temp${i}.py $i $TOTAL_BATCHES"
    done
    COMMAND="${COMMAND}; exec bash"

    case "$(uname)" in
        "Linux")
            gnome-terminal -- bash -c "$COMMAND" &
            ;;
        "Darwin")
            osascript -e "tell app \"Terminal\" to do script \"$COMMAND\"" &
            ;;
    esac
    echo "Running command for kernel $k: $COMMAND"
done
