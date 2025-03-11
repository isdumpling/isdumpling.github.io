#!/bin/bash
set -e

# 确保在主仓库中提交最新更改（关键！）
git add .
git commit -m "提交最新文章: $(date +'%Y-%m-%d %H:%M')" || true  # 允许空提交

# 保留 Hugo 生成的 GitInfo 数据
hugo --config hugo.yaml --cleanDestinationDir --minify --enableGitInfo

# 仅同步 public 目录（不重置 Git 历史）
cd public
git init
git checkout -b main

# 添加远端仓库（保留历史）
if ! git remote | grep -q origin; then
  git remote add origin git@github.com:isdumpling/isdumpling.github.io.git
fi

# 拉取旧提交（避免强制覆盖）
git pull origin main --allow-unrelated-histories || true

# 提交新更改（保留时间线）
git add .
git commit -m "Auto update: $(date +'%Y-%m-%d %H:%M')"
git push origin main

echo "✅ 部署成功！访问地址：https://isdumpling.github.io"
