import subprocess

cmd = 'cd ~/Documents/dolphin/build/Binaries && ./dolphin-emu --no-python-subinterpreters\
    --script /home/tyler/Documents/WiiRL/FelkFork/dolphinScriptTest.py\
    --exec="/home/tyler/Documents/GameCollection/Wii Play (Europe) (En,Fr,De,Es,It).nkit.gcz"'

subprocess.call(cmd, shell=True)
