const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle,
  ImageRun, PageBreak, PageNumber, Footer, Header, TabStopType, TabStopPosition,
  UnderlineType,
} = require("docx");

const PAGE_W = 12240, PAGE_H = 15840; // US Letter, DXA

function h1(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_1, spacing: { before: 320, after: 160 } });
}
function h2(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_2, spacing: { before: 260, after: 120 } });
}
function p(text, opts = {}) {
  return new Paragraph({
    spacing: { after: 160, line: 276 },
    alignment: opts.center ? AlignmentType.CENTER : AlignmentType.JUSTIFIED,
    children: [new TextRun({ text, italics: opts.italics, bold: opts.bold, size: opts.size })],
  });
}
function pMixed(runs, opts = {}) {
  return new Paragraph({
    spacing: { after: 160, line: 276 },
    alignment: opts.center ? AlignmentType.CENTER : AlignmentType.JUSTIFIED,
    children: runs,
  });
}
function cite(nums) {
  return new TextRun({ text: `[${nums.join(",")}]`, superScript: false });
}

function makeCell(text, opts = {}) {
  return new TableCell({
    width: { size: opts.width || 2000, type: WidthType.DXA },
    shading: opts.header ? { type: ShadingType.CLEAR, fill: "1F4E79" } : undefined,
    verticalAlign: "center",
    children: [new Paragraph({
      alignment: opts.center !== false ? AlignmentType.CENTER : AlignmentType.LEFT,
      children: [new TextRun({ text: String(text), bold: opts.header, color: opts.header ? "FFFFFF" : undefined, size: 20 })],
    })],
  });
}

function table(headers, rows, widths) {
  const colWidths = widths || headers.map(() => Math.floor(9000 / headers.length));
  return new Table({
    width: { size: 9000, type: WidthType.DXA },
    columnWidths: colWidths,
    rows: [
      new TableRow({
        tableHeader: true,
        children: headers.map((hd, i) => makeCell(hd, { header: true, width: colWidths[i] })),
      }),
      ...rows.map(r => new TableRow({
        children: r.map((c, i) => makeCell(c, { width: colWidths[i] })),
      })),
    ],
  });
}

function figure(path, widthPx, heightPx, maxWidthInches, caption) {
  // docx-js ImageRun transformation width/height are in PIXELS at 96 DPI.
  const targetPx = maxWidthInches * 96;
  const scale = targetPx / widthPx;
  const w = Math.round(widthPx * scale);
  const hgt = Math.round(heightPx * scale);
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 200, after: 80 },
      children: [new ImageRun({ type: "png", data: fs.readFileSync(path), transformation: { width: w, height: hgt } })],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 240 },
      children: [new TextRun({ text: caption, italics: true, size: 19 })],
    }),
  ];
}

const IMG_MAX_W = 6.3; // inches, fits within 6.5in content width (US Letter, 1in margins)

const doc = new Document({
  styles: {
    default: { document: { run: { font: "Times New Roman", size: 22 } } },
  },
  sections: [
    // ---------------- Title page / front matter ----------------
    {
      properties: { page: { size: { width: PAGE_W, height: PAGE_H }, margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
      headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [new TextRun({ text: "Meva & Popat — Apple Leaf Disease Ablation Study", size: 16, italics: true })] })] }) },
      footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], size: 18 })] })] }) },
      children: [
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 240 },
          children: [new TextRun({ text: "Does Added Model Complexity Help on a Near-Saturated Benchmark?", bold: true, size: 32 })],
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 360 },
          children: [new TextRun({ text: "An Empirical Ablation Study of Attention, Focal Loss, and Physically-Inspired Augmentation for Apple Leaf Disease Classification on PlantVillage", bold: true, size: 26 })],
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 60 },
          children: [new TextRun({ text: "Divyakant Meva\u00B9, Kalpesh Popat\u00B9", size: 22 })],
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 40 },
          children: [new TextRun({ text: "\u00B9 Faculty of Computer Applications, Marwadi University, Rajkot, Gujarat, India", italics: true, size: 20 })],
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 360 },
          children: [new TextRun({ text: "Corresponding author: Divyakant Meva", italics: true, size: 20 })],
        }),

        h2("Abstract"),
        p("Recent literature on apple leaf disease classification with the PlantVillage dataset reports consistent gains from combining transfer-learning backbones with attention mechanisms, focal loss, class weighting, and physically-inspired data augmentation. We designed a controlled ablation study to test whether these design choices, applied without a pretrained backbone, improve classification performance over a plain convolutional baseline on the same benchmark. Using the apple leaf subset of PlantVillage (3,171 images across four classes: Apple Scab, Black Rot, Cedar Apple Rust, and healthy), we trained a plain CNN baseline and three increasingly sophisticated variants incorporating squeeze-and-excitation channel attention, focal loss, class weighting, physically-inspired augmentation, test-time augmentation, and a two-model ensemble, all trained from scratch under identical computational constraints. Contrary to the pattern reported in prior transfer-learning studies, none of the added-complexity variants outperformed the plain baseline: the baseline achieved 99.37% test accuracy and 0.9928 macro-F1 (3 misclassifications out of 479 test images), while the best variant (a baseline-plus-attention ensemble) reached 99.16% accuracy and 0.9910 macro-F1. Qualitative inspection of the baseline's three errors shows they stem from atypical lighting and faint, early-stage lesions rather than architectural limitations. We argue that this apple-leaf benchmark is close to saturated for lab-condition images, leaving little headroom for loss-function or attention-based interventions to demonstrate benefit, and that the value of such techniques reported elsewhere is likely contingent on either harder (field-condition) data or genuine pretrained backbones \u2014 neither of which this from-scratch ablation was able to test. We release all code, splits, trained-model logs, and figures for reproducibility.", { italics: true }),
        pMixed([new TextRun({ text: "Keywords: ", bold: true, italics: true, size: 22 }), new TextRun({ text: "apple leaf disease classification; PlantVillage; ablation study; attention mechanism; focal loss; class imbalance; data augmentation; convolutional neural network", italics: true, size: 22 })]),
      ],
    },

    // ---------------- Main body ----------------
    {
      properties: { page: { size: { width: PAGE_W, height: PAGE_H }, margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
      headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [new TextRun({ text: "Meva & Popat — Apple Leaf Disease Ablation Study", size: 16, italics: true })] })] }) },
      footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], size: 18 })] })] }) },
      children: [
        h1("1. Introduction"),
        p("Apple leaf disease classification using deep learning has matured rapidly on the PlantVillage benchmark, with reported test accuracies ranging from roughly 93% to above 99.7% across a wide variety of architectures and training strategies [1\u201314]. A recurring theme in this literature is that the PlantVillage apple subset exhibits a moderate class imbalance (Apple Scab, Black Rot, Cedar Apple Rust, and healthy leaves occur at an approximate ratio of 1:1:0.4:2.6), and that a substantial share of recent work addresses this imbalance through a combination of pretrained transfer-learning backbones, attention mechanisms, focal loss, class-weighted objectives, and increasingly \u201cphysically-inspired\u201d augmentation strategies such as CLAHE-based contrast enhancement, illumination simulation, and sensor-noise injection, in place of, or alongside, classical oversampling [1,2,6,7,8,9]."),
        p("Fidan et al. [2] provide the most directly relevant precedent for the present study: in a controlled ablation across three data-balancing scenarios and four architectures, they found that the benefit of a given balancing strategy is architecture-dependent rather than universal \u2014 physically-inspired augmentation improved YOLOv11x but not EfficientNet-B1, whose optimum was reached on the imbalanced baseline data without additional intervention. This finding motivated the present study's central question: on the specific case of the PlantVillage apple leaf subset, do attention mechanisms, focal loss, class weighting, and physically-inspired augmentation improve classification performance over a plain convolutional baseline, when neither the baseline nor the augmented variant has access to a pretrained backbone?"),
        p("We emphasize the pretrained-backbone caveat because it materially separates this study from most of the transfer-learning literature it draws on. Ashmafee et al. [3], Moza and Wijaya [11], and Ali et al. [9], among others, report accuracies of 98.7\u201399.8% using pretrained EfficientNetV2, Inception-ResNet-V2, and EfficientNet-B0 backbones respectively, fine-tuned on this same apple subset. Our training environment had no route to pretrained ImageNet weights, so every model in this study \u2014 baseline and \u201cproposed\u201d alike \u2014 was trained from randomly initialized weights. This is a genuine limitation for benchmarking against the transfer-learning literature, but it isolates a narrower and still useful question: holding the backbone fixed (from-scratch, modest capacity), do the auxiliary design choices \u2014 attention, loss re-weighting, and augmentation \u2014 pay for themselves?"),
        p("Our answer, based on the experiments below, is: not on this dataset, at this scale. We report the full ablation, including the specific images the baseline misclassifies, and discuss why we believe this is a genuine property of the benchmark rather than an artifact of implementation choices."),

        h1("2. Related Work"),
        h2("2.1 Class imbalance handling"),
        p("Sulistyowati et al. [1] applied SMOTE oversampling to a VGG16 classifier on an imbalanced (1:5) apple leaf dataset, improving accuracy from 85.16% to 92.94%. Ding et al. [8] combined a dual attention mechanism with focal loss and class-balancing in RFCA ResNet, reporting substantial gains over a standard CNN baseline on a complex-background apple dataset. Fidan et al. [2] is the closest methodological precedent to this work: they benchmark physically-inspired augmentation (CLAHE, illumination simulation, sensor noise) against resampling-based balancing across four architectures, finding effects that are architecture- and dataset-dependent rather than uniformly positive \u2014 a finding our results are consistent with."),
        h2("2.2 Transfer learning and attention"),
        p("The majority of high-accuracy results on this specific benchmark rely on pretrained backbones. Ashmafee et al. [3] and Nazir et al. [4] both use pretrained EfficientNetV2-family backbones with runtime augmentation, reaching 99.21% and 96.21% respectively. Banjar et al. [7] (E-AppleNet) combine a pretrained EfficientNetV2 backbone with an added attention mechanism and multi-class focal loss, reaching 99% accuracy. Assad et al. [12] explicitly compare a pretrained ResNet-50 backbone against a baseline trained from scratch and report a 21.5 percentage-point mean accuracy gap in favor of the pretrained model \u2014 direct evidence, from a related apple-disease dataset, that the backbone (not the auxiliary loss/augmentation choices) may be the dominant factor in these reported gains."),
        h2("2.3 Lightweight from-scratch models"),
        p("A smaller body of work trains compact CNNs from scratch on this benchmark. Vishnoi et al. [5] and Raj et al. [10] both report approximately 98% accuracy using lightweight from-scratch CNNs on the same 3,171-image apple subset used here, with Raj et al. [10] using an identical stratified split protocol. Li et al. [13] report 99.8% validation accuracy with a lightweight RegNet variant, also without full pretrained-weight transfer. These results establish that from-scratch models are a credible baseline on this dataset, consistent with the 99.37% accuracy we observe for our own plain baseline."),
        h2("2.4 Data augmentation strategies"),
        p("Banu and Arefin [6] compare MixUp and CutMix augmentation on a pretrained MobileNet backbone, finding CutMix raises validation accuracy from 94.6% to 98.1% and specifically reduces minority-class misclassifications. Ali et al. [9] combine class weighting, stratified splitting, and augmentation with a fine-tuned EfficientNet-B0, reaching 99.78% on the PlantVillage apple subset. These results, like Fidan et al. [2], motivated the specific augmentation and class-weighting choices tested in our v1 and v3 variants below."),

        h1("3. Materials and Methods"),
        h2("3.1 Dataset"),
        p("We used the apple leaf disease subset of the PlantVillage dataset, obtained from the public mirror at github.com/spMohanty/PlantVillage-Dataset (raw/color split), comprising 3,171 laboratory-condition RGB images across four classes: Apple Scab (630), Black Rot (621), Cedar Apple Rust (275), and healthy (1,645). This is the same subset and image count used by Raj et al. [10] and Moza and Wijaya [11], with a class imbalance ratio (healthy : Cedar Apple Rust) of approximately 6:1."),
        h2("3.2 Data splitting"),
        p("Images were split 70/15/15 into training (2,218), validation (474), and test (479) sets using stratified sampling per class, matching the protocol reported by Moza and Wijaya [11]. The split was fixed (seed = 42) and identical across all model variants to ensure a fair comparison; no image appeared in more than one split."),
        h2("3.3 Baseline architecture"),
        p("The baseline is a plain convolutional neural network: four convolutional blocks (32\u2192256 channels, each block consisting of two 3\u00D73 convolutions with batch normalization and ReLU activation, followed by 2\u00D72 max pooling), global average pooling, dropout (p=0.4), and a linear classification head. The baseline is trained with standard cross-entropy loss and only light geometric augmentation (random horizontal/vertical flip, \u00B125\u00B0 rotation). Input resolution is 64\u00D764 pixels. Total trainable parameters: 1,174,244."),
        h2("3.4 Proposed variants"),
        p("We evaluated three variants intended to test the design recommendations synthesized from the literature review in Section 2:"),
        p("\u2022 v1 (\u201cAttentiveFocalNet\u201d): identical backbone to the baseline, with squeeze-and-excitation (SE) channel attention added after each convolutional block; trained with focal loss (\u03B3=2) and inverse-square-root class weighting; augmented with a heavier, \u201cphysically-inspired\u201d pipeline (CLAHE contrast enhancement, gamma/illumination jitter, additive Gaussian sensor noise, geometric flips/rotation) and an additional 3\u00D7 oversampling multiplier on the minority class (Cedar Apple Rust) via re-augmented copies."),
        p("\u2022 v3: the same SE-attention backbone, run at a higher input resolution (96\u00D796 px) with a deliberately lighter, more targeted augmentation pipeline (mild CLAHE, \u00B112\u00B0 rotation, mild brightness jitter only \u2014 the aggressive noise/rotation from v1 was removed), trained with label-smoothed (\u03B5=0.1), class-weighted cross-entropy in place of focal loss, and evaluated with test-time augmentation (TTA: averaging the softmax output of each test image with that of its horizontal mirror)."),
        p("\u2022 Ensemble: the softmax outputs of the trained baseline and v3 models, averaged per test image."),
        h2("3.5 Training environment and constraints"),
        p("All models were trained from randomly initialized weights on a single CPU core (no GPU) with approximately 4 GB of available RAM. Critically, the training environment had no network route to common pretrained-weight hosts (e.g., download.pytorch.org, huggingface.co), so transfer learning \u2014 a component of the majority of high-accuracy results cited in Section 2 \u2014 could not be tested. This is the single most important limitation of this study and is discussed further in Section 6. All models were trained with the Adam optimizer (initial learning rate 1e-3, weight decay 1e-4), a plateau learning-rate scheduler on validation macro-F1, and early stopping (patience 6\u20137 epochs) on validation macro-F1, up to a maximum of 25 epochs."),
        h2("3.6 Evaluation protocol"),
        p("All models are evaluated once, after training, on the same held-out 479-image test set, which was not used for any model selection, early stopping, or hyperparameter decisions during training (those used the validation split only). We report accuracy, macro-averaged F1, weighted F1, macro-averaged one-vs-rest AUC, per-class precision/recall/F1, and the full confusion matrix."),

        h1("4. Results"),
        h2("4.1 Training dynamics"),
        p("Figure 1 shows validation macro-F1 and validation loss across training epochs for the baseline and the v1 proposed model. The baseline exhibits a pronounced instability at epoch 8 (validation loss spikes to 0.93; macro-F1 drops to 0.735) before recovering, whereas the v1 model's validation curve is markedly smoother throughout training, despite converging to a similar or slightly lower final value. This is the clearest positive signal we found for the attention/focal-loss/class-weighting combination: it did not improve final accuracy, but it did stabilize optimization."),
        ...figure("/home/claude/training_curves.png", 1800, 750, IMG_MAX_W, "Figure 1. Validation macro-F1 (left) and validation loss (right) per epoch, baseline vs. v1 proposed model."),
        h2("4.2 Test-set performance"),
        p("Table 1 summarizes test-set performance for all four model configurations. The plain baseline achieved the highest accuracy, macro-F1, and weighted-F1 of any configuration tested. The ensemble of baseline and v3 was the closest competitor but did not exceed the baseline alone on any aggregate metric."),
        table(
          ["Model", "Accuracy", "Macro-F1", "Weighted-F1", "Macro-AUC", "Errors / 479"],
          [
            ["Baseline (plain CNN)", "99.37%", "0.9928", "0.9937", "0.9997", "3"],
            ["v1 \u2013 SE-attn + focal loss + class wt. + heavy aug.", "98.54%", "0.9855", "0.9854", "0.9994", "7"],
            ["v3 \u2013 SE-attn + light aug. + label smoothing + TTA", "98.75%", "0.9864", "0.9874", "0.9990", "6"],
            ["Ensemble (baseline + v3, averaged softmax)", "99.16%", "0.9910", "0.9916", "0.9998", "4"],
          ],
          [3200, 1400, 1400, 1500, 1300, 1200]
        ),
        p("", { size: 4 }),
        h2("4.3 Per-class analysis"),
        p("Table 2 and Figure 2 break performance down by class for the baseline and the v1 proposed model. Performance on the minority class, Cedar Apple Rust (42 test images), was identical between the baseline and v1 (precision 1.000, recall 0.976, F1 0.988 for both) \u2014 the additional minority-class oversampling and class weighting in v1 did not change the outcome on this class."),
        table(
          ["Class (n)", "Baseline P / R / F1", "v1 P / R / F1", "v3 P / R / F1"],
          [
            ["Apple Scab (95)", "0.989 / 0.989 / 0.989", "0.978 / 0.958 / 0.968", "1.000 / 0.947 / 0.973"],
            ["Black Rot (94)", "1.000 / 1.000 / 1.000", "1.000 / 1.000 / 1.000", "0.989 / 1.000 / 0.995"],
            ["Cedar Apple Rust (42)", "1.000 / 0.976 / 0.988", "1.000 / 0.976 / 0.988", "1.000 / 0.976 / 0.988"],
            ["Healthy (248)", "0.992 / 0.996 / 0.994", "0.980 / 0.992 / 0.986", "0.980 / 1.000 / 0.990"],
          ],
          [2400, 2200, 2200, 2200]
        ),
        p("", { size: 4 }),
        ...figure("/home/claude/per_class_f1.png", 1200, 750, IMG_MAX_W * 0.7, "Figure 2. Per-class test F1-score, baseline vs. v1 proposed model."),
        ...figure("/home/claude/confusion_matrices.png", 1800, 750, IMG_MAX_W, "Figure 3. Test-set confusion matrices for the baseline (left) and v1 proposed model (right)."),
        h2("4.4 Error analysis"),
        p("The baseline model misclassified exactly 3 of 479 test images: a healthy leaf predicted as Apple Scab, an Apple Scab leaf predicted as healthy, and a Cedar Apple Rust leaf predicted as healthy. Figure 4 shows all three. Visual inspection suggests none is an obvious model failure: the first (left) has an atypical purple-toned background and shadow across part of the leaf; the second (center) shows only very faint, early-stage scab lesions; the third (right) shows only a few small, pale rust spots. We interpret these as instances near the intrinsic decision boundary of the task, rather than errors a different architecture would straightforwardly resolve."),
        ...figure("/home/claude/misclassified.png", 600, 200, IMG_MAX_W * 0.8, "Figure 4. The three test images misclassified by the baseline model. Left: true healthy, predicted Apple Scab. Center: true Apple Scab, predicted healthy. Right: true Cedar Apple Rust, predicted healthy."),

        h1("5. Discussion"),
        p("The central finding of this study is negative with respect to our starting hypothesis: none of the attention, focal-loss, class-weighting, augmentation, TTA, or ensembling interventions tested improved on the plain from-scratch baseline's test accuracy or macro-F1 on this dataset. We consider three explanations."),
        p("First, the benchmark itself is close to saturated for lab-condition images. With only 3 errors out of 479 test images for the simplest model tested, there is very little room for any intervention to demonstrate a measurable improvement; a single flipped prediction changes macro-F1 by roughly 0.002\u20130.006 depending on which class is affected. This is consistent with a broader observation in the plant-disease literature that PlantVillage's uniform lab background and single-leaf framing make it substantially easier than field-condition imagery, where complex backgrounds and overlapping symptoms are known to depress accuracy considerably."),
        p("Second, our qualitative error analysis (Section 4.4) shows the baseline's few remaining errors are attributable to atypical lighting and faint, early-stage symptoms rather than confusable disease patterns between classes \u2014 precisely the kind of error that attention mechanisms and re-weighted losses are not designed to fix, since they operate on label frequency and feature salience, not image-level ambiguity or annotation-boundary cases."),
        p("Third, and most importantly, the literature results this study set out to test \u2014 particularly Ashmafee et al. [3], Banjar et al. [7], and Ali et al. [9], all reporting 99%+ accuracy with attention/focal-loss/augmentation combined with a pretrained backbone \u2014 may owe most of their gains to the pretrained backbone itself rather than the auxiliary techniques. Assad et al.'s [12] direct from-scratch-versus-pretrained comparison (a 21.5-point accuracy gap) is suggestive evidence for this on a related apple dataset. Our study could not test this directly because of the sandboxed training environment's lack of access to pretrained weights, but the pattern in our results \u2014 where attention and loss re-weighting alone, without a pretrained backbone, bought training stability but not accuracy \u2014 is consistent with the hypothesis that the backbone, not the auxiliary techniques, is doing most of the work in the wider literature."),
        p("We note this is consistent with, not contradictory to, Fidan et al.'s [2] own conclusion that balancing-strategy benefit is architecture- and dataset-dependent rather than universal. Our results extend that conclusion to a second axis: the benefit may also be backbone-dependent, appearing primarily when a strong pretrained feature extractor is present to exploit the more discriminative training signal that attention and re-weighted losses provide."),

        h1("6. Limitations and Threats to Validity"),
        p("(1) No pretrained backbone. This is the most significant limitation; all comparisons in this study are from-scratch-vs-from-scratch, and cannot speak to whether attention/focal-loss/augmentation would help when layered on top of a pretrained EfficientNet, ResNet, or Inception backbone as in the majority of the cited literature."),
        p("(2) Small test set. At 479 images, single-image differences move macro-F1 by roughly 0.002\u20130.006; the gaps we observed between the four configurations (0.0018\u20130.0073 macro-F1) are on the same order as this granularity, so we would not characterize the ranking among the non-baseline variants as strongly distinguishable from one another, only that none clearly exceeded the baseline."),
        p("(3) Single training run per configuration. Given compute constraints (single CPU core), we did not repeat training across multiple random seeds; reported numbers reflect one run per configuration and do not include variance estimates. A bagging or multi-seed ensemble is a natural extension (see Section 7)."),
        p("(4) Resolution and capacity. All models used a compact 4-block CNN at 64\u201396 px resolution, substantially smaller and lower-resolution than typical pretrained-backbone pipelines (which commonly use 224\u2013300 px inputs). This may understate the ceiling achievable by the attention/augmentation techniques at higher capacity."),
        p("(5) Dataset scope. Results are specific to the PlantVillage apple subset (lab-condition, single-leaf, uniform background) and should not be extrapolated to field-condition apple leaf imagery without further testing."),

        h1("7. Future Work"),
        p("The most direct extension is to re-run this exact ablation with a pretrained backbone (e.g., EfficientNetV2-S or ResNet-50, as used in [3,7,9]) in an environment with internet access to pretrained weights, to test whether the attention/focal-loss/augmentation techniques show a benefit when a stronger feature extractor is present. A second extension is multi-seed bagging of the baseline architecture (3\u20135 seeds, averaged predictions) to test whether variance-reduction alone can close the small remaining gap. Third, applying the same ablation protocol to a field-condition apple leaf dataset (e.g., in the spirit of FieldPlant or Plant Pathology 2021) would test whether our saturation hypothesis holds \u2014 we would predict a larger benefit from attention and augmentation on harder, more visually ambiguous data."),

        h1("8. Conclusion"),
        p("On the PlantVillage apple leaf disease subset, a plain convolutional baseline trained from scratch achieved 99.37% test accuracy and 0.9928 macro-F1, and was not outperformed by any of three progressively more sophisticated from-scratch variants incorporating SE channel attention, focal loss, class weighting, physically-inspired augmentation, test-time augmentation, or ensembling. The baseline's few remaining errors appear attributable to atypical lighting and faint early-stage symptoms rather than systematic architectural weaknesses. We interpret this as evidence that, on this specific near-saturated benchmark and in the absence of a pretrained backbone, the auxiliary design choices commonly bundled with transfer-learning pipelines in the literature do not by themselves provide additional discriminative power \u2014 their reported benefit elsewhere is more plausibly attributable to the pretrained backbone, to harder data, or to both. We release all code and results to support replication and extension of this ablation, particularly with pretrained backbones."),

        h1("Author Contributions"),
        p("Following the CRediT (Contributor Roles Taxonomy) framework:"),
        pMixed([new TextRun({ text: "Divyakant Meva: ", bold: true }), new TextRun({ text: "Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data curation, Writing \u2013 original draft, Writing \u2013 review & editing, Visualization, Supervision, Project administration." })]),
        pMixed([new TextRun({ text: "Kalpesh Popat: ", bold: true }), new TextRun({ text: "Conceptualization, Methodology, Validation, Formal analysis, Writing \u2013 review & editing, Resources." })]),
        p("Both authors have read and approved the final manuscript."),

        h1("Conflicts of Interest"),
        p("The authors declare no conflict of interest."),

        h1("Data Availability"),
        p("The apple leaf subset of the PlantVillage dataset used in this study is publicly available at https://github.com/spMohanty/PlantVillage-Dataset. All code, trained-model training logs, evaluation results, and figures generated in this study are available in the accompanying code repository."),

        h1("Code Availability"),
        p("All source code (data splitting, model definitions, training scripts, and evaluation scripts) is provided in the accompanying repository under the MIT License."),

        h1("References"),
        p("[1] Sulistyowati, T., Purwanto, P., Alzami, F., & Pramunendar, R. A. (2023). VGG16 deep learning architecture using imbalance data methods for the detection of apple leaf diseases. Moneter: Jurnal Keuangan dan Perbankan. https://doi.org/10.32832/moneter.v11i1.57"),
        p("[2] Fidan, E., Aksoy, S., Demircio\u011Flu, P., & Bogrekci, I. (2026). Class imbalance-aware deep learning approach for apple leaf disease recognition. AgriEngineering. https://doi.org/10.3390/agriengineering8020070"),
        p("[3] Ashmafee, M. H., Ahmed, T., Ahmed, S., Hasan, M. B., Jahan, M. N., & Rahman, A. (2023). An efficient transfer learning-based approach for apple leaf disease classification. 2023 International Conference on Electrical, Computer and Communication Engineering (ECCE). https://doi.org/10.1109/ecce57851.2023.10101542"),
        p("[4] Nazir, M. I., Ullah, M. O., & Akter, A. (2023). An efficient deep transfer learning based apple leaf disease classification. 2023 26th International Conference on Computer and Information Technology (ICCIT). https://doi.org/10.1109/iccit60459.2023.10441087"),
        p("[5] Vishnoi, V. K., Kumar, K., Kumar, B., Mohan, S., & Khan, A. (2023). Detection of apple plant diseases using leaf images through convolutional neural network. IEEE Access. https://doi.org/10.1109/access.2022.3232917"),
        p("[6] Banu, A., & Arefin, S. (2025). Apple leaf disease recognition via MobileNet and data augmentation: MixUp and CutMix. 2025 IEEE 7th International Conference on Sustainable Technologies For Industry 5.0 (STI). https://doi.org/10.1109/sti69347.2025.11367601"),
        p("[7] Banjar, A., Javed, A., Nawaz, M., & Dawood, H. (2025). E-AppleNet: An enhanced deep learning approach for apple fruit leaf disease classification. Applied Fruit Science. https://doi.org/10.1007/s10341-024-01239-w"),
        p("[8] Ding, J., Zhang, C., Cheng, X., Yue, Y., Fan, G., Wu, Y., & Zhang, Y. (2023). Method for classifying apple leaf diseases based on dual attention and multi-scale feature extraction. Agriculture. https://doi.org/10.3390/agriculture13050940"),
        p("[9] Ali, H., Shifa, N., Benlamri, R., Farooque, A., & Yaqub, R. (2025). A fine tuned EfficientNet-B0 convolutional neural network for accurate and efficient classification of apple leaf diseases. Scientific Reports. https://doi.org/10.1038/s41598-025-04479-2"),
        p("[10] Raj, T., Singh, S., Belwal, P., & Chauhan, S. (2025). An optimized CNN model for accurate and fast apple leaf disease classification. 2025 IEEE International Conference on Computer, Electronics, Electrical Engineering & their Applications (IC2E3). https://doi.org/10.1109/ic2e365635.2025.11166678"),
        p("[11] Moza, E., & Wijaya, N. (2026). Application of transfer learning using Inception-ResNet-V2 for image-based classification of apple leaf diseases. Green Intelligent Systems and Applications. https://doi.org/10.53623/gisa.v6i1.1165"),
        p("[12] Assad, A., Bhat, M., Bhat, Z. A., Ahanger, A. N., Kundroo, M., Dar, R., Ahanger, A. B., & Dar, B. N. (2023). Apple diseases: detection and classification using transfer learning. Quality Assurance and Safety of Crops & Foods. https://doi.org/10.15586/qas.v15isp1.1167"),
        p("[13] Li, L., Zhang, S., & Wang, B. (2021). Apple leaf disease identification with a small and imbalanced dataset based on lightweight convolutional networks. Sensors. https://doi.org/10.3390/s22010173"),
        p("[14] Doutoum, A., & Tugrul, B. (2025). A systematic review of deep learning techniques for apple leaf diseases classification and detection. PeerJ Computer Science. https://doi.org/10.7717/peerj-cs.2655"),
      ],
    },
  ],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync("/home/claude/manuscript.docx", buf);
  console.log("Wrote /home/claude/manuscript.docx");
});
