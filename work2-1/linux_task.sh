#任务一：1.在当前目录下创建一个名为 linux_practice的文件夹，其内部包含两个子目录:docs 和 backup。
mkdir -p linux_practice/docs
mkdir -p linux_practice/backup

#2.在 docs 目录下创建三个文件:readme.txt、notes.l0g 和 temp.tmp。
touch linux_practice/docs/readme.txt
touch linux_practice/docs/notes.log
touch linux_practice/docs/temp.tmp

#3.删除 temp.tmp 文件，将 notes.0g 重命名为 daily_report.txt
rm -f linux_practice/docs/temp.tmp
mv linux_practice/docs/notes.log linux_practice/docs/daily_report.txt

#重命名其他实现
#先复制再删除原文件
#cp linux_practice/docs/notes.log linux_practice/docs/daily_report.txt
#rm -f linux_practice/docs/notes.log

#4.向 daily report.txt写入第一行内容:"Proiect Status: Active”追加第二行内容，显示当前系统日期(使用 date 命令)
echo "Project Status: Active"> linux_practice/docs/daily_report.txt
date >> linux_practice/docs/daily_report.txt

#5.将 docs 目录下的所有 .txt 文件复制到 backup 目录下。
cp linux_practice/docs/*.txt linux_practice/backup/

#6.将 backup 目录下所有文件的权限修改为 只读 -r--r--r-
chmod 444 linux_practice/backup/*

#修改完成后，脚本需自动输出如下信息:Archive Complete.File[文件名]is now read-only.
for file in linux_practice/backup/*; do
      echo "Archive Complete.File[$(basename "$file")]is now read-only."
  done
