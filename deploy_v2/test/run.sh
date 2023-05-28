#!/usr/bin/env bash
# 测试shell脚本，这个脚本会被main py调用，并且该脚本还会运行test py脚本

for i in {1..10}
do
  python3 test.py
done
