#!/bin/bash
set -e

# 强制清理并重新生成
rm -rf public
hugo --config hugo.yaml --cleanDestinationDir --minify

# 进入 public 前重置 Git 环境
cd public
rm -rf .git  # 关键！删除残留的 Git 配置
git init
git checkout -b main
git add .
git commit -m "Auto update: $(date +'%Y-%m-%d %H:%M')"
git remote add origin git@github.com:isdumpling/isdumpling.github.io.git
git push -f origin main

echo "✅ 部署成功！访问地址：https://isdumpling.github.io"
