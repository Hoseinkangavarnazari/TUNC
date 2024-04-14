#!/bin/bash

# Usage example:      ./start.sh ./pr_new.py 1 5 10 3
#                     (This will run 1 to 5 batches in 3 kernels with a total of 10 batches)

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
TEMP_SCRIPT_DIR="${DIRPATH}/temp_script"
TEMP_PR_DIR="${DIRPATH}/temp_pr"  # Path for the temp_pr directory

# Create temp_script and temp_pr directories
mkdir -p "$TEMP_SCRIPT_DIR"
mkdir -p "$TEMP_PR_DIR"

for ((k=0; k<KERNEL; k++))
do
    SCRIPT_FILE="${TEMP_SCRIPT_DIR}/run_kernel_${k}.sh"
    echo "#!/bin/bash" > "$SCRIPT_FILE"
    echo "cd $PARENT_DIR" >> "$SCRIPT_FILE"
    case "$(uname)" in
        "Linux")
            echo "source ./venvDA/bin/activate && cd $DIRPATH" >> "$SCRIPT_FILE"
            ;;
        "Darwin")
            echo "source ./venvDip/bin/activate && cd $DIRPATH" >> "$SCRIPT_FILE"
            ;;
    esac

    for ((i=START_BATCH+k; i<=END_BATCH; i+=KERNEL))
    do
        cp "$FILENAME" "${TEMP_PR_DIR}/temp${i}.py"  # Copy to temp_pr directory
        echo "python3 ${TEMP_PR_DIR}/temp${i}.py $i $TOTAL_BATCHES && wait" >> "$SCRIPT_FILE"
    done
    echo "exec bash" >> "$SCRIPT_FILE"

    # Make the script executable
    chmod +x "$SCRIPT_FILE"

    case "$(uname)" in
        "Linux")
            gnome-terminal -- bash -c "$SCRIPT_FILE" &
            ;;
        "Darwin")
            osascript -e "tell app \"Terminal\" to do script \"$SCRIPT_FILE\"" &
            ;;
    esac
    echo "Running script for kernel $k: $SCRIPT_FILE"
done
