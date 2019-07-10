#!/bin/bash

if [ $# -ne 2 ]; then
	echo "Usage: <this script> <Remote Public URL> <Remote Port>"
	exit
fi

echo "Forwarding port "$2" on "$1" to localhost:8157..."

ssh -N -L 8157:$1:$2 zliu6676@$1
