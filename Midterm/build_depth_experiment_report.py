from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "Midterm" / "RVM深度信息融合调研与实验报告.docx"
FIGURE = ROOT / "Midterm" / "DepthFusionSyntheticExperiment" / "comparison.png"


def set_cell_shading(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_borders(cell, color="D9D9D9", size="6"):
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = tc_pr.first_child_found_in("w:tcBorders")
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tc_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        tag = "w:" + edge
        element = borders.find(qn(tag))
        if element is None:
            element = OxmlElement(tag)
            borders.append(element)
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), size)
        element.set(qn("w:color"), color)


def set_cell_margins(cell, top=100, start=110, bottom=100, end=110):
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for margin, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn("w:" + margin))
        if node is None:
            node = OxmlElement("w:" + margin)
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_repeat_table_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def format_table(table, widths=None):
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    set_repeat_table_header(table.rows[0])
    for row_index, row in enumerate(table.rows):
        for col_index, cell in enumerate(row.cells):
            set_cell_borders(cell)
            set_cell_margins(cell)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            if widths:
                cell.width = Inches(widths[col_index])
            if row_index == 0:
                set_cell_shading(cell, "1F4E78")
                for paragraph in cell.paragraphs:
                    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    for run in paragraph.runs:
                        run.font.bold = True
                        run.font.color.rgb = RGBColor(255, 255, 255)
                        run.font.size = Pt(9.5)
            else:
                if row_index % 2 == 0:
                    set_cell_shading(cell, "EAF2F8")
                for paragraph in cell.paragraphs:
                    paragraph.paragraph_format.space_after = Pt(0)
                    paragraph.paragraph_format.line_spacing = 1.1
                    for run in paragraph.runs:
                        run.font.size = Pt(9.5)


def add_table(doc, headers, rows, widths):
    table = doc.add_table(rows=1, cols=len(headers))
    for i, text in enumerate(headers):
        table.rows[0].cells[i].text = text
    for row_data in rows:
        cells = table.add_row().cells
        for i, text in enumerate(row_data):
            cells[i].text = str(text)
    format_table(table, widths)
    doc.add_paragraph().paragraph_format.space_after = Pt(0)
    return table


def add_bullet(doc, text, level=0):
    paragraph = doc.add_paragraph(style="List Bullet" if level == 0 else "List Bullet 2")
    paragraph.add_run(text)
    return paragraph


NUMBER_COUNTER = 0


def add_number(doc, text, restart=False):
    global NUMBER_COUNTER
    if restart:
        NUMBER_COUNTER = 0
    NUMBER_COUNTER += 1
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.left_indent = Inches(0.24)
    paragraph.paragraph_format.first_line_indent = Inches(-0.18)
    paragraph.add_run(f"{NUMBER_COUNTER}. {text}")
    return paragraph


def add_hyperlink(paragraph, text, url):
    part = paragraph.part
    rel_id = part.relate_to(
        url,
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink",
        is_external=True,
    )
    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), rel_id)
    run = OxmlElement("w:r")
    r_pr = OxmlElement("w:rPr")
    color = OxmlElement("w:color")
    color.set(qn("w:val"), "0563C1")
    underline = OxmlElement("w:u")
    underline.set(qn("w:val"), "single")
    r_pr.append(color)
    r_pr.append(underline)
    run.append(r_pr)
    text_node = OxmlElement("w:t")
    text_node.text = text
    run.append(text_node)
    hyperlink.append(run)
    paragraph._p.append(hyperlink)


def add_page_number(paragraph):
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = "PAGE"
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    text = OxmlElement("w:t")
    text.text = "1"
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.extend([begin, instr, separate, text, end])


def set_font(run, name="Microsoft YaHei", size=None, bold=None, color=None):
    run.font.name = name
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), name)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.font.bold = bold
    if color is not None:
        run.font.color.rgb = RGBColor(*color)


def configure_styles(doc):
    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Microsoft YaHei"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    normal.font.size = Pt(10.5)
    normal.font.color.rgb = RGBColor(0, 0, 0)
    normal.paragraph_format.line_spacing = 1.35
    normal.paragraph_format.space_after = Pt(6)

    title = styles["Title"]
    title.font.name = "Microsoft YaHei"
    title._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    title.font.size = Pt(24)
    title.font.bold = True
    title.font.color.rgb = RGBColor(0, 0, 0)

    for style_name, size in (("Heading 1", 16), ("Heading 2", 13), ("Heading 3", 11.5)):
        style = styles[style_name]
        style.font.name = "Microsoft YaHei"
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.paragraph_format.space_before = Pt(12)
        style.paragraph_format.space_after = Pt(6)
        style.paragraph_format.keep_with_next = True

    for style_name in ("List Bullet", "List Bullet 2", "List Number"):
        style = styles[style_name]
        style.font.name = "Microsoft YaHei"
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
        style.font.size = Pt(10.5)


def build_document():
    doc = Document()
    doc.core_properties.title = "RVM 深度信息融合调研与实验报告"
    doc.core_properties.subject = "固定大屏前演员分割中的深度先验融合"
    doc.core_properties.author = ""
    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(0.75)
    section.bottom_margin = Inches(0.7)
    section.left_margin = Inches(0.8)
    section.right_margin = Inches(0.8)
    configure_styles(doc)

    # Cover
    spacer = doc.add_paragraph()
    spacer.paragraph_format.space_after = Pt(68)
    title = doc.add_paragraph(style="Title")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.add_run("RVM 深度信息融合调研与实验报告")
    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.paragraph_format.space_before = Pt(14)
    run = subtitle.add_run("固定大屏前演员完整分割方案")
    set_font(run, size=15, bold=False, color=(0, 0, 0))
    meta = doc.add_paragraph()
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    meta.paragraph_format.space_before = Pt(80)
    meta.add_run("项目  RVM 前景分割优化\n")
    meta.add_run("报告日期  2026 年 11 月 10 日\n")
    meta.add_run("代码目录  D:\\CODES\\RVM")
    for run in meta.runs:
        set_font(run, size=11, color=(70, 70, 70))
    doc.add_page_break()

    # Header/footer after cover; Word applies to whole section, intentionally subtle.
    header_p = section.header.paragraphs[0]
    header_p.text = "RVM 深度信息融合调研与实验报告"
    header_p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    for run in header_p.runs:
        set_font(run, size=8.5, color=(100, 100, 100))
    add_page_number(section.footer.paragraphs[0])
    for run in section.footer.paragraphs[0].runs:
        set_font(run, size=8.5, color=(100, 100, 100))

    doc.add_heading("报告结论", level=1)
    doc.add_paragraph(
        "深度信息可以加入当前 RVM 链路。现阶段最稳妥的方式不是改变 RVM 的网络输入和权重，而是在 RVM 输出 Alpha 后加入固定背景深度模型，以高置信度深度证据抑制背景泄漏并补回人物内部缺失区域。该方案已经接入现有离线流程，并保持无深度参数时的原始行为。"
    )
    doc.add_paragraph(
        "受控合成实验中，Alpha 平均绝对误差由 0.0752 降至 0.0193，相对下降 74.4%，且无效深度区域与原 RVM Alpha 的最大差异为 0。该结果证明融合逻辑和失效回退机制有效，但不代表现场相机的最终质量；仓库当前没有与 RGB 同步配准的真实深度序列，因此仍需使用真实 RGB-D 素材完成最终评测。"
    )
    add_table(
        doc,
        ["项目", "结果", "说明"],
        [
            ["实现状态", "已完成", "深度背景建模、Alpha 融合、诊断图和命令行参数已接入"],
            ["兼容性", "保持", "不提供深度参数时继续使用原 RVM 输出"],
            ["单元测试", "3 项通过", "前景补回、背景抑制、无效深度回退和尺寸检查"],
            ["合成实验", "MAE 下降 74.4%", "用于验证机制，不等同于真实现场结论"],
            ["真实实验", "待补充", "需要同步、配准、同名的演员深度和无人背景深度"],
        ],
        [1.25, 1.25, 4.1],
    )

    doc.add_heading("一 任务背景与目标", level=1)
    doc.add_paragraph(
        "现场存在一个稳定的空间先验：演员始终在固定大屏前方活动。大屏显示内容可能变化，但屏幕表面的空间位置不变。因此，深度能够提供比 RGB 颜色更稳定的前景证据，尤其适合处理白衣与亮背景相似、屏幕内容复杂、光照变化和背景误检等问题。"
    )
    doc.add_paragraph("本阶段工作包括以下三项内容：")
    add_number(doc, "调研能够利用深度信息进行人体分割或抠像的现有算法。", restart=True)
    add_number(doc, "结合当前 RVM 工程，确定深度的表示方式、融合位置和失效处理。")
    add_number(doc, "实现深度融合模块，并通过可重复的受控实验验证效果。")

    doc.add_heading("二 现有方法调研", level=1)
    doc.add_paragraph(
        "调研结果表明，RGB-D 人像抠图和固定背景抠图已有明确研究路线，但能够直接替换当前 RVM 且提供完整代码、权重和实时接口的方案仍不成熟。硬件 SDK 可以直接输出人体掩码，但通常不能提供发丝级软 Alpha。"
    )
    add_table(
        doc,
        ["方法", "深度用法", "可直接使用性", "对本项目的价值"],
        [
            ["DART", "背景深度统计、贝叶斯前景概率、RGB-D 细化和深度 trimap", "论文完整，官方仓库当前无实现与权重", "与固定大屏场景最匹配，适合作为设计依据"],
            ["RGB-D Human Matting", "RGB 与深度联合预测人像 Alpha", "HDM-2K 数据公开，完整部署代码不足", "证明深度有助于真实人像抠图和固定场景泛化"],
            ["ZED Body Tracking", "深度与人体检测结合，输出二维人物掩码", "有 ZED 硬件即可调用", "可直接作为人体位置或深度先验，边缘不等同于软 Alpha"],
            ["Azure Kinect Body Index", "为每个深度像素提供人体实例编号", "有对应硬件和 SDK 即可调用", "适合快速生成硬掩码或种子区域"],
            ["DFormer", "统一或双路 RGB-D 特征编码", "代码和模型可用，但任务是语义分割", "可作为后续训练 RGB-D RVM 的结构参考"],
            ["Background Matting V2", "使用无演员背景 RGB 图，不使用深度", "代码和模型成熟", "可作为固定背景 RGB baseline，但不能利用空间前后关系"],
        ],
        [1.05, 2.0, 1.55, 2.0],
    )

    doc.add_heading("三 当前算法与数据条件", level=1)
    doc.add_heading("三一 当前 RVM 链路", level=2)
    doc.add_paragraph(
        "当前工程使用 rvm_mobilenetv3_fp16.onnx。直播和离线预处理均只向模型输入 RGB 图像、四级循环状态和 downsample ratio；模型输出前景颜色、Alpha 和新的循环状态。Composure 流程读取 EXR 的 R、G、B 通道，将 RVM Alpha 与背景图合成。MatAnyone 2 质量基线同样只接收 RGB 视频和首帧掩码。"
    )
    flow = doc.add_paragraph()
    flow.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = flow.add_run("RGB 帧  →  RVM ONNX  →  前景与 Alpha  →  背景合成")
    set_font(run, size=11.5, bold=True, color=(31, 78, 120))

    doc.add_heading("三二 数据缺口", level=2)
    doc.add_paragraph(
        "现有 Results 目录未发现真实深度序列。抽查 EXR 后，文件为浮点 RGBA 画面，当前代码也只读取 RGB。真实深度实验必须补充两类数据：演员表演期间的同步深度，以及相机和屏幕位置不变时采集的多帧无人背景深度。两类深度都必须注册到制作 RGB 的像素坐标系。"
    )

    doc.add_heading("四 深度表示与融合设计", level=1)
    doc.add_heading("四一 背景深度模型", level=2)
    doc.add_paragraph(
        "对每个像素位置，从多帧无人背景深度中计算中位数和稳健离散度。中位数用于表示屏幕或固定场景的典型距离，MAD 经 1.4826 缩放后用于估计该位置的深度噪声。无效值、零值和非有限值不参与统计；有效样本不足的位置标记为不可用。"
    )
    doc.add_paragraph(
        "当前帧深度小于背景深度时，表示该像素更靠近相机。系统以背景深度减当前深度作为前景证据，并根据背景噪声自适应放宽阈值。默认按毫米工作：背景容差 30，可靠前景距离差 100，噪声倍数 3。参数需要根据实际传感器和拍摄距离重新标定。"
    )

    doc.add_heading("四二 保守融合规则", level=2)
    add_bullet(doc, "高置信度背景区域抑制 RVM 产生的背景泄漏。")
    add_bullet(doc, "高置信度前景内部补回 RVM 漏掉的人体区域。")
    add_bullet(doc, "深度边界保留 5 至 6 像素不确定带，继续使用 RVM Alpha，避免硬切和配准误差。")
    add_bullet(doc, "深度无效、背景统计无效或尺寸不匹配时不盲目修改 Alpha；尺寸不匹配直接报错。")
    add_bullet(doc, "融合强度可配置，默认不改变 RVM 模型、权重和循环状态。")
    flow2 = doc.add_paragraph()
    flow2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = flow2.add_run("RGB → RVM Alpha ┐\n                ├→ 置信度融合 → 最终 Alpha\nDepth → 背景差分 ┘")
    set_font(run, size=11, bold=True, color=(31, 78, 120))

    doc.add_heading("五 代码修改", level=1)
    add_table(
        doc,
        ["文件", "修改内容", "兼容性"],
        [
            ["depth_guided_matting.py", "新增深度读取、背景模型、同名帧匹配、边界置信度和 Alpha 融合", "独立模块，不改变 ONNX"],
            ["Composure.py", "run_rvm_batch 增加可选深度目录、背景深度目录、尺度、阈值和诊断输出", "参数缺省时走原始 RVM"],
            ["run_depth_fusion_synthetic_experiment.py", "生成受控 RGB Alpha、深度、背景帧并计算误差", "可重复运行"],
            ["test_depth_guided_matting.py", "验证前景补回、背景抑制、无效深度回退和尺寸检查", "3 项测试通过"],
        ],
        [1.75, 3.45, 1.4],
    )
    doc.add_paragraph(
        "原始 RVM 代码和 ONNX 模型已单独备份到 Backup_Original_RVM_20261110。当前深度模块不修改模型权重，回退时只需不提供 depth dir 和 background depth dir 参数。"
    )

    doc.add_heading("六 受控合成实验", level=1)
    doc.add_heading("六一 实验设置", level=2)
    doc.add_paragraph(
        "由于仓库没有真实深度，本次使用 640 × 360 的合成场景验证融合机制。目标 Alpha 包含身体、头部、双臂和细发丝；模拟 RGB Matte 加入宽光晕、人物内部缺口和远离人物的背景误检。背景深度设为约 4000 mm，人物深度设为约 2250 mm，并加入高斯噪声。背景模型由 8 帧深度建立；人物边界中约 55% 的像素被设为无效深度，用于测试保守回退。"
    )
    add_table(
        doc,
        ["参数", "取值"],
        [
            ["图像尺寸", "640 × 360"],
            ["背景深度", "4000 mm，标准差约 6 mm"],
            ["人物深度", "2250 mm，标准差约 10 mm"],
            ["背景建模帧数", "8"],
            ["背景容差", "30 mm"],
            ["可靠前景距离差", "100 mm"],
            ["深度边界保护宽度", "6 像素"],
        ],
        [2.6, 4.0],
    )

    doc.add_heading("六二 定量结果", level=2)
    add_table(
        doc,
        ["指标", "RVM 模拟 Alpha", "深度融合 Alpha", "变化"],
        [
            ["平均绝对误差 MAE", "0.075208", "0.019262", "下降 74.4%"],
            ["无效深度回退最大误差", "不适用", "0.000000", "完全保持原 Alpha"],
            ["自动化测试", "不适用", "3 项通过", "实现符合预期"],
        ],
        [2.25, 1.45, 1.45, 1.45],
    )
    doc.add_paragraph(
        "结果表明，深度证据能够消除远离人物的背景误检，并补回人物内部缺失区域；边界深度缺失时仍保留 RGB Matte 的柔和轮廓。误差下降主要来自已知构造缺陷被深度先验纠正，因此只能证明实现正确和方案具有潜力。"
    )
    if FIGURE.exists():
        paragraph = doc.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        picture = paragraph.add_run().add_picture(str(FIGURE), width=Inches(6.85))
        doc_pr = picture._inline.docPr
        doc_pr.set("descr", "合成实验中真实 Alpha 模拟 RGB Alpha 深度前景证据与融合结果的对比")
        caption = doc.add_paragraph()
        caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
        caption.paragraph_format.keep_with_next = False
        run = caption.add_run("图 1  合成实验对比  从左到右为目标 Alpha 模拟 RGB Matte 深度证据和融合结果")
        set_font(run, size=9, color=(80, 80, 80))

    doc.add_heading("七 局限性与风险", level=1)
    add_bullet(doc, "合成实验不能覆盖真实深度相机的飞点、多径干扰、运动模糊和曝光同步问题。")
    add_bullet(doc, "LED 大屏可能影响主动红外或 ToF 深度相机，需现场测量无效深度比例。")
    add_bullet(doc, "深度相机与制作相机存在视差；配准误差会首先出现在头发、手指和快速运动轮廓。")
    add_bullet(doc, "相机移动会使逐像素背景模板失效；此时需利用相机位姿重投影，或每帧重新拟合屏幕平面。")
    add_bullet(doc, "透明、反光和黑色材质可能没有可靠深度，因此必须保留 RVM 回退通道。")
    add_bullet(doc, "当前方案会保留位于屏幕前的道具；若只需要人体，需要再叠加人体检测或实例掩码。")

    doc.add_heading("八 真实 RGB-D 实验方案", level=1)
    doc.add_paragraph("真实效果评估建议按以下顺序执行：")
    add_number(doc, "固定制作相机、深度相机和大屏，完成深度到 RGB 的标定与像素级配准。", restart=True)
    add_number(doc, "采集至少 30 帧无人背景深度，建立中位数和 MAD 背景模型。")
    add_number(doc, "同步采集白衣、亮背景、发丝、快速手部动作、手持道具和多人遮挡片段。")
    add_number(doc, "分别运行原始 RVM、深度单独阈值、RVM 加深度融合三组方案。")
    add_number(doc, "标注代表性帧 Alpha，计算 MAE、MSE、Gradient、Connectivity 和时序 dtSSD，并记录 FPS、延迟和无效深度率。")
    add_number(doc, "对背景容差、前景距离差、边界宽度和融合强度做消融实验，确定现场参数。")
    doc.add_paragraph(
        "验收重点应同时覆盖人物完整率和背景泄漏率。若后处理融合仍不能满足发丝质量，再考虑引入深度 trimap 加 ViTMatte，或训练独立深度编码器与 RVM 多尺度特征进行中期融合。"
    )

    doc.add_heading("九 结论", level=1)
    doc.add_paragraph(
        "本次工作确认深度信息能够加入当前算法，并完成了不改动 RVM 权重的深度引导 Alpha 融合实现。该方案充分利用演员位于固定大屏前方的空间先验，兼顾可回退性和工程风险。受控实验验证了背景抑制、前景补回和无效深度回退，但现场收益仍取决于深度质量与 RGB 配准精度。下一步应优先采集真实 RGB-D 数据并完成参数标定，而不是立即训练新的四通道 RVM。"
    )

    doc.add_heading("参考资料", level=1)
    references = [
        ("DART  Depth Enhanced Accurate and Real Time Background Matting", "https://arxiv.org/abs/2402.15820"),
        ("DART 官方仓库", "https://github.com/Fenghoo/DART"),
        ("RGB-D Human Matting 与 HDM-2K", "https://github.com/rgbd-zml/RGB-D-human-matting"),
        ("ZED Body Tracking API", "https://docs.stereolabs.com/docs/development/zed-sdk/modules/body-tracking/using-the-api"),
        ("Azure Kinect Body Index Map", "https://microsoft.github.io/Azure-Kinect-Body-Tracking/release/1.1.x/class_microsoft_1_1_azure_1_1_kinect_1_1_body_tracking_1_1_frame_ae056039bd8d86a5e8dbdc5a2defafeae.html"),
        ("DFormer RGB-D Semantic Segmentation", "https://github.com/VCIP-RGBD/DFormer"),
        ("Background Matting V2", "https://github.com/PeterL1n/BackgroundMattingV2"),
        ("Robust Video Matting", "https://github.com/PeterL1n/RobustVideoMatting"),
    ]
    for index, (name, url) in enumerate(references, 1):
        paragraph = doc.add_paragraph()
        paragraph.paragraph_format.left_indent = Inches(0.18)
        paragraph.paragraph_format.first_line_indent = Inches(-0.18)
        paragraph.paragraph_format.space_after = Pt(2)
        run = paragraph.add_run(f"[{index}] {name}  ")
        set_font(run, size=9)
        add_hyperlink(paragraph, "访问来源", url)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    build_document()
