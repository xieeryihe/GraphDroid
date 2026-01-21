该项目也是本人的研究生项目，实验所用到的开源数据集在 resources/数据集.xlsx 中。

baseline方面，鉴于大部分代码难以直接运行，因此本人对其代码进行了简单改造，在不影响原始逻辑的情况下，增加了便于记录和运行的相关内容。

- DroidBot: https://github.com/xieeryihe/DroidBot.git
- RLDroid: https://github.com/xieeryihe/RLDroid-run.git
  - Effectively Modeling UI Transition Graphs for Android Apps via Reinforcement Learning (ICPC 2025)
- Gator: https://github.com/xieeryihe/gator-run.git
  - Static Window Transition Graphs for Android（2018）
- GoalExplorer: 只能在linux中运行
  - 作者有打包好的jar包可以直接用 https://github.com/resess/GoalExplorer/releases/tag/v1.2.2
  - 这里给出能跑的指令：`java -jar ./GoalExplorer.jar ge -i K9.apk -s /xxx/Android/SDK -o output` 其中 GoalExplorer.jar 就是下载的jar包名字，-i是要分析的安装包，-s 的是安卓SDK目录，需要自己下载SDK。

A3E方法（Targeted and depth-first exploration for systematic testing of android apps）实在难以运行，作者提供的虚拟机巨卡，而且是老古董级别的Android版本，建议直接放弃作baseline。
