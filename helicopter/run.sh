#!/bin/sh

cd trainers/consoleTrainerJavaHelicopter
bash run.bash >&1 &
cd -
cd agents/helicopterAgentCPP
bash run.bash
cd -