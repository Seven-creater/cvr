# Audio-CVR Reference Verification

Verified: 2026-07-24

This audit follows the `nature-ref-verifier` principle: metadata are checked
against primary publisher or conference pages rather than copied from secondary
bibliographies. ArXiv is retained only where no more authoritative archival
record is used.

| Key | Venue record checked | Metadata status | Authoritative source |
|---|---|---|---|
| `ventura2024covr` | AAAI 2024 | Authors, title, volume 38(6), pages 5270--5279, DOI verified | https://ojs.aaai.org/index.php/AAAI/article/view/28334 |
| `ventura2024covr2` | IEEE TPAMI 2024 | Authors, title, volume 46(12), pages 11409--11421 verified | https://ieeexplore.ieee.org/document/10685001 |
| `hummel2024egocvr` | ECCV 2024 | Authors, title, venue, year verified | https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/5363_ECCV_2024_paper.php |
| `liu2021cirr` | ICCV 2021 | Authors, title, pages 2125--2134 verified | https://openaccess.thecvf.com/content/ICCV2021/html/Liu_Image_Retrieval_on_Real-Life_Images_With_Pre-Trained_Vision-and-Language_Models_ICCV_2021_paper.html |
| `ji2026omnicvr` | ICLR 2026 | Authors, title, archival venue and year verified | https://iclr.cc/virtual/2026/poster/10010075 |
| `han2026cova` | ICASSP 2026 | Authors, title, archival venue and year verified | https://www.cmsworkshops.com/ICASSP2026/view_paper.php?PaperNum=9901&bare=1 |
| `chen2026e5omni` | Findings of ACL 2026 | Authors, title, pages 19430--19443 and DOI verified | https://aclanthology.org/2026.findings-acl.970/ |
| `girdhar2023imagebind` | CVPR 2023 | Authors, title, pages 15180--15190 verified | https://openaccess.thecvf.com/content/CVPR2023/html/Girdhar_ImageBind_One_Embedding_Space_To_Bind_Them_All_CVPR_2023_paper.html |
| `jiang2024vlm2vec` | arXiv 2410.05160 | Authors, title, identifier and year verified | https://arxiv.org/abs/2410.05160 |
| `chu2024qwen2audio` | arXiv 2407.10759 | Authors, title, identifier and year verified | https://arxiv.org/abs/2407.10759 |
| `jeong2025avigate` | CVPR 2025 | Authors, title, pages 26202--26211 verified | https://openaccess.thecvf.com/content/CVPR2025/html/Jeong_Learning_Audio-guided_Video_Representation_with_Gated_Attention_for_Video-Text_Retrieval_CVPR_2025_paper.html |
| `chen2020vggsound` | ICASSP 2020 | Authors, title, venue and year verified | https://www.robots.ox.ac.uk/~vgg/data/vggsound/ |
| `tian2018ave` | ECCV 2018 | Authors, title, venue and year verified | https://arxiv.org/abs/1803.08842 |

## Claim-Level Checks

- CoVR/CoVR-2 support prior automatic CVR data construction; Audio-CVR does not
  claim that automatic construction itself is new.
- OmniCVR already places the source in its retrieval setting; Audio-CVR claims
  exact source masking and source-specific diagnosis, not first source inclusion.
- CoVA already studies audio-visual composed video retrieval; Audio-CVR claims a
  narrower audio-primary directional diagnostic.
- CoVA reports that direct trimodal fusion can underperform V+T while learned
  selective fusion recovers the benefit; AVIGATE independently motivates
  gating because blindly incorporated audio can be uninformative.
- ImageBind supports a fixed zero-shot multimodal embedding baseline. Its
  negative audio-fusion result in this paper is reported rather than tuned away.
- The VLM2Vec experiment uses the public VLM2Vec-Qwen2VL-7B checkpoint with
  Qwen2-Audio captions and is explicitly labeled as a reproduction, not as the
  unavailable official AudioVLM2Vec checkpoint.

No unresolved author, title, venue, year, page, or DOI conflict remains in the
main-paper bibliography.
