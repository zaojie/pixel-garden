# 雨庭 · Pixel Garden

用 **Ink Studio** 制作的可交互三维像素庭院：樱花、神社、石灯、池塘和小桥。通过 SDF 雕刻形体，烘焙为高斯点，再由原生 WebGL 2 渲染器绘制。

**[在线体验](https://zaojie.github.io/pixel-garden/)** · [AI 项目作品集](https://github.com/zaojie)

![Ink Studio 像素风庭院](docs/ink-pixel-preview.jpg)

## 交互

- 拖动旋转、滚轮缩放，或用方向键调整视角、`+` / `-` 缩放。
- 切换「像素风 / 原始高斯」，比较同一份高斯数据的两种渲染效果。
- 调整像素颗粒、抖动强度，开启自动环绕，暂停雨滴与落花。
- 页面底部可打开早期 Canvas 点渲染原型作比较。

## 制作流程

`SDF 造型与布尔运算 → S.bake() 表面高斯点烘焙 → Ink Studio WebGL 2 → 色板、轮廓与像素后处理`

当前场景有 **72,304 个高斯点**，其中 **72,091 个来自 SDF 表面烘焙**。每个原生高斯点包含位置、密度、颜色和协方差；雨滴与落花使用 Ink Studio 的 `KIT.weather.fall`。这是一份程序生成的场景，不是照片训练重建，也不是原视频资产。

![关闭像素后处理的原始高斯模式](docs/ink-raw-preview.jpg)

## 本地运行

需要 Python 3 和支持 WebGL 2 的浏览器；直接运行已烘焙场景无需 npm 安装。

```sh
git clone https://github.com/zaojie/pixel-garden.git
cd pixel-garden
python3 -m http.server 8765 --bind 127.0.0.1
```

打开 <http://127.0.0.1:8765/>。

修改 `ink-garden.mjs` 后，使用 Node.js 18+ 重新烘焙：

```sh
node bake-ink.mjs
```

烘焙脚本同时校验点数量、数据长度和几何字段的有效性。

## 主要文件

| 文件 | 用途 |
| --- | --- |
| `ink-garden.mjs` | SDF 场景、材质、灯光和天气定义 |
| `bake-ink.mjs` | 烘焙、检查、输出场景数据 |
| `ink-view.mjs` | 官方渲染器与页面交互 |
| `ink-garden.bin` / `.json` | 原生高斯数据与场景元信息 |
| `ink-toolkit.mjs` | 官方 bake profile 生成的工具包 |
| `vendor/ink-studio/` | 未修改的官方渲染器及其依赖 |

## 致谢与许可

使用 [Silicon Jungle 的 Ink Studio](https://github.com/siliconjungle/shapeshift-labs-ink-studio)，固定提交 `cbda532f6fe21a5429b4ea94b917cf53b8c3ca18`。工具包及官方渲染器采用 MIT 许可，原始许可保留在 [vendor/ink-studio/LICENSE](vendor/ink-studio/LICENSE)。

仅打包运行所需模块，不包含上游完整仓库。`ink-toolkit.mjs` 由该提交的 `studio.bundle({profile:'bake', format:'module'})` 生成。
