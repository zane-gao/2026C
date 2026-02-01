
这份文档为您详细说明了补充数据集（Enriched Data）相对于原题数据的新增内容、数据意义以及在建模中的潜在价值。

### 2026 MCM Problem C 补充数据集说明文档

#### 1. 数据集概述

原数据集（`2026_MCM_Problem_C_Data.csv`）主要提供了选手（名人）与舞伴的基本信息（姓名、行业、家乡、年龄等）以及各赛季、各周次的裁判评分和比赛结果。

 **补充数据集** （`2026美赛C题补充数据集！.xlsx - enriched.csv`）在保持原有所有列不变的基础上， **新增了 10 列关于社交媒体影响力的数据** 。这些数据涵盖了每一位名人（Celebrity）及其专业舞伴（Professional Partner）在主流社交平台上的粉丝数量。

#### 2. 新增数据内容详解

新增的 10 个字段具体如下（数值代表粉丝/订阅者数量）：

**A. 名人（Celebrity）社交媒体数据**

* `celebrity_instagram_followers`: 名人在 Instagram 上的粉丝数。
* `celebrity_twitter_followers`: 名人在 Twitter $X$ 上的粉丝数。
* `celebrity_tiktok_followers`: 名人在 TikTok 上的粉丝数。
* `celebrity_youtube_subscribers`: 名人在 YouTube 上的订阅人数。
* `celebrity_total_followers_wikidata`: 基于 Wikidata 记录的名人全网粉丝总数估算值。

**B. 专业舞伴（Partner）社交媒体数据**

* `partner_instagram_followers`: 专业舞伴在 Instagram 上的粉丝数。
* `partner_twitter_followers`: 专业舞伴在 Twitter $X$ 上的粉丝数。
* `partner_tiktok_followers`: 专业舞伴在 TikTok 上的粉丝数。
* `partner_youtube_subscribers`: 专业舞伴在 YouTube 上的订阅人数。
* `partner_total_followers_wikidata`: 基于 Wikidata 记录的专业舞伴全网粉丝总数估算值。

> **注意** ：部分选手的某些平台数据可能为空（或为0），这代表该选手未在该平台开设账号或数据缺失。

#### 3. 数据意义

根据赛题背景描述，比赛结果由 **裁判评分** （Judge Scores）和 **观众投票** （Fan Votes）共同决定。

* **原数据局限性** ：原数据仅提供了“裁判评分”，而“观众投票”是未知且保密的（unknown and a closely guarded secret）。题目明确指出观众投票受到“名人的受欢迎程度和个人魅力（popularity and charisma）”的显著影响。
* **新数据意义** ：新增的社交媒体数据是**“人气”和“粉丝基础”的直接量化指标**。它填补了模型中关于“观众缘”这一关键缺失变量的空白，为推断未知的观众投票提供了最强有力的外部证据。

#### 4. 对建模的潜在价值

这部分新数据对于解决 Problem C 的核心任务具有极高的价值，具体体现在以下几个方面：

**1. 估算观众投票（Fan Vote Estimation）—— 最核心价值**

* **作为代理变量（Proxy Variable）** ：题目要求建立模型估算未知的观众投票。社交媒体粉丝数可以作为一个强有力的代理变量。逻辑是：粉丝基数越大的选手，潜在的观众票仓越大。
* **构建先验概率** ：在贝叶斯框架下，粉丝数可以用来构建选手获得观众投票的“先验概率”。
* **解释“不合理”的晋级** ：对于那些裁判评分低但未被淘汰的“争议”选手（如题目提到的 Bobby Bones 或 Bristol Palin），新数据可以验证是否是因为他们拥有巨大的社交媒体粉丝基数（即“场外因素”压倒了“技术因素”）。

**2. 分析“舞伴效应”（Partner Impact Analysis）**

* 题目要求分析“专业舞伴”对比赛结果的影响。
* 新数据提供了舞伴的粉丝数。这意味着模型可以分析：**一个自带流量的“明星舞伴”是否能拯救一个默默无闻或跳舞很差的名人？** 这为评估舞伴价值提供了除舞蹈技术指导之外的另一个维度（引流能力）。

**3. 细化人气来源（Platform Specificity）**

* 不同行业的名人可能在不同平台占优势（例如 TikTok 可能对年轻网红更有利，Twitter 可能对政客或老牌明星更有利）。
* 通过分析不同平台的粉丝数与比赛排名的相关性，可以研究 DWTS 的核心观众群体更偏向于哪种类型的社交媒体用户。

**4. 提升预测模型的准确性**

* 在构建二分类模型（预测本周是否淘汰 `Eliminated`）时，加入粉丝数特征通常能显著提高模型的 AUC 或准确率，因为这解释了裁判分数无法解释的那部分方差。

### 总结

这份补充数据将原本纯粹基于“技术得分”的内部视角，扩展到了包含“社会影响力”的外部视角。对于解决 **Problem C** 中关于“估算观众投票”、“解释评分与结果的差异（争议）”以及“分析选手特征影响”等任务，这部分数据是至关重要的“解题钥匙”。
