#!/bin/bash
set -e

# 主仓库提交（保留原始仓库结构）
git add .
git commit -m "提交最新文章: $(date +'%-%m-%d %H:%M')" || true

# 强制清空public目录（但保留.git历史）
rm -rf public/*
[ -d public/.git ] && mv public/.git .public_git_backup

# 带缓存清理的Hugo构建
hugo --config hugo.yaml --cleanDestinationDir --minify --enableGitInfo --gc

# 恢复Git历史记录
if [ -d .public_git_backup ]; then
    mv .public_git_backup public/.git
fi

# 进入发布目录
cd public

# 智能初始化Git仓库
if [ ! -d .git ]; then
    git init
    git remote add origin git@github.com:isdumpling/isdumpling.github.io.git
    git checkout -b main
fi

# 确保拉取策略安全
git fetch origin
git reset --hard origin/main || git reset --hard ORIG_HEAD

# 提交并强推（仅限GitHub Pages场景）
git add --all --force
git commit -m "Auto update: $(date +'%Y-%m-%d %H:%M')" --allow-empty
git push origin main -f

echo "✅ 部署成功！访问地址：https://isdumpling.github.io"
