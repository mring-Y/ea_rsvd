# GitHub 与 Zenodo 上传说明

## 包的用途

本次发布是当前稿件的源码与实验数值证据快照。原创代码采用 MIT 许可，
第三方许可证单独保留。压缩包包含代码、测试、实验配置、来源记录和经过
清理的 CSV/JSON 证据；矩阵原始文件、完整因子、构建产物与运行日志不在包内。
矩阵来源和下载方式保存在数据说明与来源清单中。

运行 `python scripts/prepare_release.py` 后，输出目录包括：

| 文件或目录 | 用途 |
| --- | --- |
| `github/ea_het_rsvd_repro/` | 可交给 Git 或 GitHub Desktop 提交的仓库目录 |
| `ea_het_rsvd_repro_20261017_github.zip` | GitHub 源码目录的便携副本，解压后提交文件 |
| `ea_het_rsvd_repro_20261017_zenodo.zip` | 直接上传到 Zenodo 的源码与证据归档 |
| `*.zip.sha256` | 对应压缩包的 SHA-256 校验值 |
| `package_report.json` | 包大小、文件数和打包检查结果 |

两个 ZIP 来自同一个快照，内容一致。一个用于解压后提交 GitHub，另一个
用于在 Zenodo 归档；不需要把两个 ZIP 同时上传到 Zenodo。

## GitHub

建议使用 GitHub Desktop 或 Git 提交整个 `github/ea_het_rsvd_repro/` 目录。
网页一次选择超过 100 个文件会触发文件数量限制，即使总大小很小也可能失败。
通过 Git 提交不受该网页选择器的限制。

若从 ZIP 开始，请先解压，把仓库根目录中的 `README.md`、`LICENSE`、
`src/`、`scripts/`、`benchmarks/` 等作为仓库内容提交。仅上传 ZIP 会使
访客无法直接浏览和检索其中的代码。命令行示例见 [publishing.md](publishing.md)。

发布目录没有携带原目录的 `.git`。作者姓名、单位与版权署名是公开引用信息，
有意保留在 `CITATION.cff` 和 `LICENSE` 中。未来提交产生的 Git 作者信息
由你本机的 Git 配置决定；如需隐藏私人邮箱，可使用 GitHub 的 noreply 邮箱。

## Zenodo

如果继续维护此前同一软件项目，优先在原记录上选择创建新版本，将本次
Zenodo ZIP 放入新版本草稿。检查草稿继承的旧文件，确保新版本使用本次
归档。旧版本的文件与 DOI 应保留，不能把旧 DOI 当成本次快照的 DOI。

建议元数据：

- 类型：Software。
- 标题：Residual Pair Diagnosis and Finite Grid Calibration for Sparse GPU Randomized SVD — Source and Numerical Evidence。
- 作者：Ziming Yuan；单位以 `CITATION.cff` 为准。
- 许可：MIT；随包的第三方材料仍按各自许可证。
- 当前快照内部版本：`1.6.0-dev`，与本包 `CITATION.cff` 一致。
- 描述：提供稀疏 GPU 随机 SVD 的残差分量诊断、同起点候选动作比较、有限网格成本校准，以及数值验证和跨设备对照的复现代码与筛选后的实验记录。

`10.5281/zenodo.23262787` 是此前实验归档的 DOI。本次准备包尚无自己的
新 DOI，也未上传。取得新版本草稿分配或预留的 DOI 后，将其写入论文及
`CITATION.cff`，同步调整发布状态与版本说明，并重新生成归档和校验值，
再完成发布。若另建独立记录，应在相关标识符中关联此前归档。

## 校验与复现

仓库根目录的 `release_manifest.json` 给出每个载荷文件的大小和 SHA-256；
`SHA256SUMS.txt` 还包含该清单自身的校验值。数值证据子目录同时保留原有
的 `file_manifest.json`。不要批量改动证据文件的换行格式，否则其字节哈希
会改变。`.gitattributes` 已配置为保留原始字节；Shell 脚本使用 LF 换行。

稿件证据的无 GPU 重建命令见
[数值证据说明](../benchmarks/manuscript_completion/README.md)。GPU 实验入口、
依赖与计时范围见 [reproducibility.md](reproducibility.md) 和各基准目录说明。
打包完整性校验不等同于重新运行全部 CUDA 实验；新设备上的性能与数值结果
仍需按相同协议实测。
