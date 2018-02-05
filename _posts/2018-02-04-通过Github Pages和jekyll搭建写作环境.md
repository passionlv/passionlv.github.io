---
layout: post
title:  "通过Github Pages和jekyll搭建写作环境"
categories: 工具
tags:  使用指南
author: xiaopeng
---

* content
{:toc}

本文记录了我第一次使用Github Pages和jekyll创建写作空间的步骤，感谢[HyG](https://github.com/Gaohaoyang/gaohaoyang.github.io) 提供了这么优秀的jekyll theme。




## jekyll使用笔记
### jekyll安装  
date: 2018-02-04 22:31:00 +0800
安装软件及命令如下：
1. ruby ： sudo apt-get install ruby
2. ruby-dev : sudo apt-get install ruby-dev
3. jekyll : sudo gem install jekyll
4. bundler : sudo gem install bunduler
5. nodejs : sudo gem install nodeis
安装软件中不包含git和python常见软件。可以通过命令加上-v看是否安装成功。

PS： Ruby 的gem类似于Python的pip。gem程序是基于ruby的组件或者库。

## 选择jekyll theme
通过网上筛选，选择：[HyG](https://github.com/Gaohaoyang/gaohaoyang.github.io)作为我搭建笔记的jekyll theme，原因如下：
1. 国人写的，方便学习
2. 功能强大：主页、摘要、归档、分类、标签、评论、统计、目录等一应俱全。当然了学习门框也提高了，使用到了XXXX技术（现在我所用技术还是小白）。
3. Markdown编写，并且支持mathjax，使得通过Latex编写公式公式很方便。（这也是我选github wiki而用github pages + jekyll的主要原因）

### 启动jekyll theme
输入如下命令：
1. 下载jekyll模板`git clone https://github.com/Gaohaoyang/gaohaoyang.github.io.git`
2. 进入到模板文件夹内
3. `jekyll s`  

如果提示：Error:  jekyll-paginate。直接按照该工具即可：sudo gem install jekyll-paginate。  
执行成功后，在浏览器中打开网页：*http://127.0.0.1:4000/*  就可以看到下载的界面了。下面将该theme定制为自己博客theme。

### 修改jekyll theme
当每次更改后，可刷新浏览器即可看到定制效果。  
#### 清空blog文件
在_posts文件夹中，将博客文章删除。建议保留一两篇作为编写的示例。 刷新浏览器即可看到
当删除博客文章后，相应的Archives、Categories、Tags页面都自动更新。主页的侧边栏同样自动更新。  
#### 修改主页
1. 在index.html中修改标题。
2. 在_config.yml中修改title， brief-intro两项内容

注意：修改完_config.yml后，刷新浏览器并不能看到修改，需要重新执行：jekyll build。
关于修改配置文件，在网上找到如下教程：
1. https://yq.aliyun.com/articles/26324  
#### 修改访问统计
1. 在[百度统计](tongji.baidu.com/)中注册用户和密码。
2. 在百度统计的***管理***页面，点击***新增网站***，添加网站域名：username.github.io；网站首页：username.github.io。
3. 增加好网站后，可以在代码获取页面，查到如下程序：
```js
 hm.src = "https://hm.baidu.com/hm.js?XXXXX";
```
问号后面字符串即为百度统计ID，将其更新到*_config.yml*文件中*baidu_tongji_id*项。
theme中同时还支持google analytic，对于我来说，百度统计足够了。
#### 修改评论
theme支持https://disqus.com/ 或 http://duoshuo.com/ 评论。但是前者被墙，后者已经关闭，因此百度后按照下面步骤增加了新的第三方评论。在网上查后选择韩国的[来必力](https://livere.com/)。注册后，获得data_uid和网页代码。首先，在jekyll中更改如下
首先在*_config.yml*文件中增加如下：
```
livere_data_uid=XXXX #来必力City版安装代码中的data-uid
```
然后，将livere网站提供的代码复制到*./_includes/comments.html*文件中。

### 编写md文件
#### 安装remarkable编辑器
1. 在官网下载deb文件
2. 运行：dpkg -i remarkable_1.62_all.deb （注意版本号可能会稍有不同）
3. sudo apt-get install -f （这个很关键，否则直接执行无法运行）
4. remarkable &   （&表示设置此进程为后台进程）
用了几天remarkable感觉预览显示有些bug，无法精确定位显示编辑位置。所有尝试下载atom，据说也是github出品。

#### 安装atom编辑器
1. 官网下载deb文件
2. 运行：dpkg -i atom-amd64.deb
3. sudo apt-get install -f （这个很关键，否则直接执行无法运行）
4. atom   （&表示设置此进程为后台进程）

关于atom编辑器，有几个使用技巧：
1. 官方文档：[Atom Flight Manual](https://flight-manual.atom.io/)
2. 快捷键：*Ctrl+Shift+P*
3. 中文拼写检查较烂，因此可以关闭拼写检查。“command+shift+p”关闭“spell check toggle” 选项。或者直接在package中将其disable掉。
4. markdown实时预览功能。在setting->package中将**markdown-preview**插件*disable*，然后在setting->install中安装**markdown-preview-plus**插件。
5. 在设置中常见**toggle**，该单位可理解为开关，选中它后切换开或关。
6. tree-view:toggle，表示打开或关闭树型目录窗口
7. 关于theme，我选择了一个最受欢迎的atom-material-ui，在setting->install搜索安装即可。

关于atom编辑器git功能，我单独说下：
1. 打开快捷：github=ctrl+8,git=ctrl+9，通过右下角的files切换两者
2. 通过github.clone命令去clone相应的文件
3. 在git窗口，一般的操作顺序是：
  - 文件修改后会出现在Unstaged Changes栏
  - 点击Stage All将修改过的文件加到Staged Changes栏
  - 如提交，在commit message中增加提交附录，然后点击下面**commit**按钮
  - 




## 上传至Github
上传之前需要注册github账号，并创建username.github.io的*repository*。关于github pages的使用，可参考[github pages官方帮助](https://help.github.com/articles/what-is-github-pages/)  
当完成上面工作后，需要进行将文件夹上传只github，才能通过http(s)://username.github.io直接访问博客。需要做如下工作：  
### 设置SSH key
首先，在本地生成ssk key：
```
$cd ～/.ssh  //检查是否已经存在ssh，如果存在可以先备份
$ssh-keygen -t rsa -C xxxxx@gmail.com #注册github时的email
```
然后，登陆github，在个人主页的右上角，找到settings链接，选择“**SSH and GPG keys**”页面，点击**New SSH key**中，在Title中设置注册的邮箱，在key中将生成的key文件（默认的位置是~/.ssh/id_rsa.pub）内容复制进来即可。通过ssh命令验证是否设置成功，如果出现下面提示表示设置成功：
```
$ssh -T git@github.com
Hi passionlv! You've successfully authenticated, but GitHub does not provide shell access.

```
### 设置并上传git仓库

进入到存放jekyll项目的文件夹，执行如下命令：
```
#建立git仓库
git init
#将项目中所有文件添加到仓库
git add .
#将add的文件commit到仓库
git commit -m “注释语句”
#将本地仓库关联到github上，如果创建错误的话，需要git remote rm origin去删除后重新创建
git remote add origin https://github.com/passionlv/notebook.git/passionlv.github.io.git
#上传代码到github，执行过后需要输入github用户名和密码
git push -u origin master

#如果有博客更新，也就是修改了_post下面文件，可以
git status
git commit -a
git push -u origin master
```
