# Ensemble logit pool — model coverage

**85 models ADDED** (in pool, self-F1 ≥ 0.70) · **56 EXCLUDED**. Slice = seed-42 3.5k held-out; see [POOL.md](POOL.md). Regenerate: `python experiments/ensemble/gen_pool_readme.py`.

## Added vs Excluded (summary)

| ✅ Added (model · self-F1) | ❌ Excluded (model · reason) |
|---|---|
| `bgem3_v2_1024` · 0.7463 | `t048_e0` · lowf1 |
| `cont_spectok` · 0.7367 | `e16b_granite_ls_depth11` · lowf1 |
| `cont_hist0` · 0.7304 | `e16b_granite_ls_depth14_v2` · lowf1 |
| `bgem3_hist0` · 0.7301 | `e35_apl` · lowf1 |
| `supcon` · 0.7207 | `router_e5s` · lowf1 |
| `t048_e1` · 0.7312 | `ceil_explore` · lowf1 |
| `t023_e1` · 0.7312 | `ceil_firststep` · lowf1 |
| `t043_e1` · 0.7167 | `ceil_striphist` · lowf1 |
| `e11_granite_tapt` · 0.7529 | `pair_grepread` · lowf1 |
| `e12soup_granite_ls_is3` · 0.7786 | `bgem3_hardex` · error |
| `e12soup_granite_ls_is2` · 0.769 | `bgem3_rdrop` · error |
| `bgem3_prune12` · 0.7391 | `qwen3_base` · error |
| `cont_prune12` · 0.7277 | `e35_fd_elr` · error |
| `soup_granite_ls` · 0.7243 | `e35_fd_ls` · error |
| `a24_anchor_full_scratch` · 0.7793 | `hardex` · error |
| `e16_qwen3_depth14_recover` · 0.7629 | `qwen3_champion_pruned` · error |
| `t043_e2` · 0.7543 | `a24_attn_k90_recover` · incompat |
| `k6_e2` · 0.7034 | `a24_attn_k90_scratch` · incompat |
| `granite_names_ls_full` · 0.773 | `a24_sal_k90_scratch` · incompat |
| `e22_aum06_champion` · 0.7763 | `a24_smoke_attn_k90_scratch` · incompat |
| `e22_pvi06_champion` · 0.7697 | `b24_drop2_recover` · incompat |
| `coreset_rt_pvi06` · 0.751 | `b24_drop2_scratch` · incompat |
| `coreset_rt_pvi15` · 0.7498 | `b24_drop5_recover` · incompat |
| `coreset_base` · 0.749 | `b24_drop5_scratch` · incompat |
| `coreset_rt_aum06` · 0.7487 | `b24_drop8_recover` · incompat |
| `coreset_rt_aum15` · 0.7487 | `b24_drop8_scratch` · incompat |
| `coreset_rt_cl` · 0.7461 | `e18_smoke_annealsoft` · incompat |
| `coreset_rt_forget` · 0.7431 | `e18_smoke_ft0.01` · incompat |
| `coreset_rt_el2n06` · 0.7416 | `e4_ffn_r512_recover_e8b` · incompat |
| `coreset_rt_cart15` · 0.7397 | `e12_granite_ls_s43` · leak |
| `coreset_rt_cart06` · 0.7349 | `e12_granite_ls_s44` · leak |
| `coreset_rt_el2n15` · 0.7346 | `e30_aum06_f0` · leak |
| `e25c_richmeta_ls_bf16` · 0.7792 | `e30_aum06_f1` · leak |
| `e25b_miseo_ls_bf16` · 0.7735 | `e30_aum06_f2` · leak |
| `e25a_miseo_repro` · 0.7692 | `e30_aum06_f3` · leak |
| `e26s_ba_a70_T2` · 0.7735 | `e30_aum06_f4` · leak |
| `e26s_t2_a70_T2` · 0.7697 | `e30_e25c_f0` · leak |
| `e26s_t1_a70_T2` · 0.7692 | `e30_e25c_f1` · leak |
| `e26s_t1_soft` · 0.7644 | `e30_e25c_f2` · leak |
| `e28_full_t019` · 0.777 | `e30_e25c_f3` · leak |
| `e28_full_t017` · 0.7745 | `e30_e25c_f4` · leak |
| `e28_full_t043` · 0.7706 | `e30_is3_f0` · leak |
| `e28_full_t023` · 0.77 | `e30_is3_f1` · leak |
| `e28_full_t048` · 0.7698 | `e30_is3_f2` · leak |
| `k6_e3` · 0.7415 | `e30_is3_f3` · leak |
| `t023_e3` · 0.7084 | `e30_is3_f4` · leak |
| `e30s_t1_a70` · 0.7754 | `prune12` · no_serialize |
| `e30s_t1_a50` · 0.7739 | `member_0_t019` · no_serialize |
| `e30s_ba_is3` · 0.7731 | `member_1_t023` · no_serialize |
| `e30s2_t1_a50` · 0.7726 | `member_2_t043` · no_serialize |
| `e30s2_ba_is3` · 0.7722 | `ft_Qwen__Qwen3-Embedding-0.6B` · no_serialize |
| `e30s2_t1_a70` · 0.7695 | `granite_tapt_mlm` · no_serialize |
| `e30s2_t1_soft` · 0.7678 | `granite_tapt_mlm_richargs` · no_serialize |
| `e30s_t1_soft` · 0.7652 | `hist0` · no_serialize |
| `e32_a3_neft` · 0.7785 | `rdrop` · no_serialize |
| `e32_b1_hdrop` · 0.778 | `v2_1024_full` · no_serialize |
| `e32_a1_fgm` · 0.7709 |  |
| `e34_c_awp` · 0.7808 |  |
| `e34_a_fgm` · 0.775 |  |
| `e34_anchor` · 0.7741 |  |
| `e34_b_pgd` · 0.7667 |  |
| `e35_ls` · 0.7502 |  |
| `e35_ce` · 0.7462 |  |
| `e35_boot` · 0.7403 |  |
| `e35_sce` · 0.7345 |  |
| `e35_gce` · 0.7184 |  |
| `e38_t001fd` · 0.7859 |  |
| `e38_t031fd` · 0.7856 |  |
| `e38_t031_elr` · 0.7855 |  |
| `e38_t070fd` · 0.7845 |  |
| `e38_t040_elr` · 0.7843 |  |
| `e38_t070_elr` · 0.7836 |  |
| `e38_t040fd` · 0.7797 |  |
| `e8a_ls_richargs_full` · 0.7801 |  |
| `e8a_granite_richargs_full` · 0.769 |  |
| `e8b_ls_qwen3_richargs_full` · 0.7657 |  |
| `e8b_qwen3_richargs_full` · 0.7643 |  |
| `e9_granite_ls` · 0.7493 |  |
| `e9_granite_wce` · 0.7429 |  |
| `e9_granite_la` · 0.7387 |  |
| `e9_granite_ce` · 0.7343 |  |
| `e9_granite_focal` · 0.7304 |  |
| `g_awp_swa` · 0.7811 |  |
| `estack_v2_tapt_richargs_ls` · 0.7771 |  |
| `e_stack_tapt_ls_richargs_full` · 0.7727 |  |

## Added — detail

| exp | model | self-F1 | serialize |
|---|---|---|---|
| E0/bge | `bgem3_v2_1024` | 0.7463 | richargs |
| E0/bge | `cont_spectok` | 0.7367 | bareact |
| E0/bge | `cont_hist0` | 0.7304 | v1 |
| E0/bge | `bgem3_hist0` | 0.7301 | v1 |
| E0/bge | `supcon` | 0.7207 | v1 |
| E1 | `t048_e1` | 0.7312 | cache |
| E1 | `t023_e1` | 0.7312 | cache |
| E1 | `t043_e1` | 0.7167 | cache |
| E11 | `e11_granite_tapt` | 0.7529 | v1 |
| E12 | `e12soup_granite_ls_is3` | 0.7786 | cache |
| E12 | `e12soup_granite_ls_is2` | 0.769 | cache |
| E12 | `bgem3_prune12` | 0.7391 | richargs |
| E12 | `cont_prune12` | 0.7277 | v1 |
| E12 | `soup_granite_ls` | 0.7243 | richargs |
| E15 | `a24_anchor_full_scratch` | 0.7793 | richargs |
| E16 | `e16_qwen3_depth14_recover` | 0.7629 | cache |
| E2 | `t043_e2` | 0.7543 | cache |
| E2 | `k6_e2` | 0.7034 | cache |
| E21 | `granite_names_ls_full` | 0.773 | cache |
| E22 | `e22_aum06_champion` | 0.7763 | cache |
| E22 | `e22_pvi06_champion` | 0.7697 | cache |
| E22 | `coreset_rt_pvi06` | 0.751 | v1 |
| E22 | `coreset_rt_pvi15` | 0.7498 | cache |
| E22 | `coreset_base` | 0.749 | v1 |
| E22 | `coreset_rt_aum06` | 0.7487 | cache |
| E22 | `coreset_rt_aum15` | 0.7487 | v1 |
| E22 | `coreset_rt_cl` | 0.7461 | v1 |
| E22 | `coreset_rt_forget` | 0.7431 | v1 |
| E22 | `coreset_rt_el2n06` | 0.7416 | v1 |
| E22 | `coreset_rt_cart15` | 0.7397 | cache |
| E22 | `coreset_rt_cart06` | 0.7349 | v1 |
| E22 | `coreset_rt_el2n15` | 0.7346 | cache |
| E25 | `e25c_richmeta_ls_bf16` | 0.7792 | cache |
| E25 | `e25b_miseo_ls_bf16` | 0.7735 | cache |
| E25 | `e25a_miseo_repro` | 0.7692 | cache |
| E26 | `e26s_ba_a70_T2` | 0.7735 | richargs |
| E26 | `e26s_t2_a70_T2` | 0.7697 | richargs |
| E26 | `e26s_t1_a70_T2` | 0.7692 | richargs |
| E26 | `e26s_t1_soft` | 0.7644 | richargs |
| E28 | `e28_full_t019` | 0.777 | richargs |
| E28 | `e28_full_t017` | 0.7745 | cache |
| E28 | `e28_full_t043` | 0.7706 | cache |
| E28 | `e28_full_t023` | 0.77 | cache |
| E28 | `e28_full_t048` | 0.7698 | cache |
| E3 | `k6_e3` | 0.7415 | cache |
| E3 | `t023_e3` | 0.7084 | cache |
| E30 | `e30s_t1_a70` | 0.7754 | richargs |
| E30 | `e30s_t1_a50` | 0.7739 | richargs |
| E30 | `e30s_ba_is3` | 0.7731 | richargs |
| E30 | `e30s2_t1_a50` | 0.7726 | richargs |
| E30 | `e30s2_ba_is3` | 0.7722 | richargs |
| E30 | `e30s2_t1_a70` | 0.7695 | richargs |
| E30 | `e30s2_t1_soft` | 0.7678 | richargs |
| E30 | `e30s_t1_soft` | 0.7652 | richargs |
| E32 | `e32_a3_neft` | 0.7785 | cache |
| E32 | `e32_b1_hdrop` | 0.778 | cache |
| E32 | `e32_a1_fgm` | 0.7709 | cache |
| E34 | `e34_c_awp` | 0.7808 | richargs |
| E34 | `e34_a_fgm` | 0.775 | richargs |
| E34 | `e34_anchor` | 0.7741 | richargs |
| E34 | `e34_b_pgd` | 0.7667 | richargs |
| E35 | `e35_ls` | 0.7502 | v1 |
| E35 | `e35_ce` | 0.7462 | v1 |
| E35 | `e35_boot` | 0.7403 | v1 |
| E35 | `e35_sce` | 0.7345 | v1 |
| E35 | `e35_gce` | 0.7184 | v1 |
| E38 | `e38_t001fd` | 0.7859 | richargs |
| E38 | `e38_t031fd` | 0.7856 | richargs |
| E38 | `e38_t031_elr` | 0.7855 | richargs |
| E38 | `e38_t070fd` | 0.7845 | richargs |
| E38 | `e38_t040_elr` | 0.7843 | richargs |
| E38 | `e38_t070_elr` | 0.7836 | richargs |
| E38 | `e38_t040fd` | 0.7797 | richargs |
| E8 | `e8a_ls_richargs_full` | 0.7801 | cache |
| E8 | `e8a_granite_richargs_full` | 0.769 | cache |
| E8 | `e8b_ls_qwen3_richargs_full` | 0.7657 | cache |
| E8 | `e8b_qwen3_richargs_full` | 0.7643 | cache |
| E9 | `e9_granite_ls` | 0.7493 | cache |
| E9 | `e9_granite_wce` | 0.7429 | cache |
| E9 | `e9_granite_la` | 0.7387 | cache |
| E9 | `e9_granite_ce` | 0.7343 | cache |
| E9 | `e9_granite_focal` | 0.7304 | cache |
| misc | `g_awp_swa` | 0.7811 | richargs |
| misc | `estack_v2_tapt_richargs_ls` | 0.7771 | cache |
| misc | `e_stack_tapt_ls_richargs_full` | 0.7727 | cache |

## Excluded — detail

| exp | model | decision | why |
|---|---|---|---|
| E0 | `t048_e0` | `drop_lowf1` | self-F1 < 0.70 (weak / specialist / broken checkpoint) |
| E16 | `e16b_granite_ls_depth11` | `drop_lowf1` | self-F1 < 0.70 (weak / specialist / broken checkpoint) |
| E16 | `e16b_granite_ls_depth14_v2` | `drop_lowf1` | self-F1 < 0.70 (weak / specialist / broken checkpoint) |
| E35 | `e35_apl` | `drop_lowf1` | self-F1 < 0.70 (weak / specialist / broken checkpoint) |
| E5 | `router_e5s` | `drop_lowf1` | self-F1 < 0.70 (weak / specialist / broken checkpoint) |
| misc | `ceil_explore` | `drop_lowf1` | self-F1 < 0.70 (weak / specialist / broken checkpoint) |
| misc | `ceil_firststep` | `drop_lowf1` | self-F1 < 0.70 (weak / specialist / broken checkpoint) |
| misc | `ceil_striphist` | `drop_lowf1` | self-F1 < 0.70 (weak / specialist / broken checkpoint) |
| misc | `pair_grepread` | `drop_lowf1` | self-F1 < 0.70 (weak / specialist / broken checkpoint) |
| E0/bge | `bgem3_hardex` | `error` | load error |
| E0/bge | `bgem3_rdrop` | `error` | load error |
| E0/bge | `qwen3_base` | `error` | load error |
| E35 | `e35_fd_elr` | `error` | load error |
| E35 | `e35_fd_ls` | `error` | load error |
| misc | `hardex` | `error` | load error |
| misc | `qwen3_champion_pruned` | `error` | load error |
| E15 | `a24_attn_k90_recover` | `skip_incompat` | incompatible arch — loads as garbage (pruned/factored/MLM/smoke) |
| E15 | `a24_attn_k90_scratch` | `skip_incompat` | incompatible arch — loads as garbage (pruned/factored/MLM/smoke) |
| E15 | `a24_sal_k90_scratch` | `skip_incompat` | incompatible arch — loads as garbage (pruned/factored/MLM/smoke) |
| E15 | `a24_smoke_attn_k90_scratch` | `skip_incompat` | incompatible arch — loads as garbage (pruned/factored/MLM/smoke) |
| E15 | `b24_drop2_recover` | `skip_incompat` | incompatible arch — loads as garbage (pruned/factored/MLM/smoke) |
| E15 | `b24_drop2_scratch` | `skip_incompat` | incompatible arch — loads as garbage (pruned/factored/MLM/smoke) |
| E15 | `b24_drop5_recover` | `skip_incompat` | incompatible arch — loads as garbage (pruned/factored/MLM/smoke) |
| E15 | `b24_drop5_scratch` | `skip_incompat` | incompatible arch — loads as garbage (pruned/factored/MLM/smoke) |
| E15 | `b24_drop8_recover` | `skip_incompat` | incompatible arch — loads as garbage (pruned/factored/MLM/smoke) |
| E15 | `b24_drop8_scratch` | `skip_incompat` | incompatible arch — loads as garbage (pruned/factored/MLM/smoke) |
| E18 | `e18_smoke_annealsoft` | `skip_incompat` | incompatible arch — loads as garbage (pruned/factored/MLM/smoke) |
| E18 | `e18_smoke_ft0.01` | `skip_incompat` | incompatible arch — loads as garbage (pruned/factored/MLM/smoke) |
| E4 | `e4_ffn_r512_recover_e8b` | `skip_incompat` | incompatible arch — loads as garbage (pruned/factored/MLM/smoke) |
| E12 | `e12_granite_ls_s43` | `skip_leak` | leak — didn't hold out the seed-42 slice (KFold fold / seed 43-44) |
| E12 | `e12_granite_ls_s44` | `skip_leak` | leak — didn't hold out the seed-42 slice (KFold fold / seed 43-44) |
| E30 | `e30_aum06_f0` | `skip_leak` | leak — didn't hold out the seed-42 slice (KFold fold / seed 43-44) |
| E30 | `e30_aum06_f1` | `skip_leak` | leak — didn't hold out the seed-42 slice (KFold fold / seed 43-44) |
| E30 | `e30_aum06_f2` | `skip_leak` | leak — didn't hold out the seed-42 slice (KFold fold / seed 43-44) |
| E30 | `e30_aum06_f3` | `skip_leak` | leak — didn't hold out the seed-42 slice (KFold fold / seed 43-44) |
| E30 | `e30_aum06_f4` | `skip_leak` | leak — didn't hold out the seed-42 slice (KFold fold / seed 43-44) |
| E30 | `e30_e25c_f0` | `skip_leak` | leak — didn't hold out the seed-42 slice (KFold fold / seed 43-44) |
| E30 | `e30_e25c_f1` | `skip_leak` | leak — didn't hold out the seed-42 slice (KFold fold / seed 43-44) |
| E30 | `e30_e25c_f2` | `skip_leak` | leak — didn't hold out the seed-42 slice (KFold fold / seed 43-44) |
| E30 | `e30_e25c_f3` | `skip_leak` | leak — didn't hold out the seed-42 slice (KFold fold / seed 43-44) |
| E30 | `e30_e25c_f4` | `skip_leak` | leak — didn't hold out the seed-42 slice (KFold fold / seed 43-44) |
| E30 | `e30_is3_f0` | `skip_leak` | leak — didn't hold out the seed-42 slice (KFold fold / seed 43-44) |
| E30 | `e30_is3_f1` | `skip_leak` | leak — didn't hold out the seed-42 slice (KFold fold / seed 43-44) |
| E30 | `e30_is3_f2` | `skip_leak` | leak — didn't hold out the seed-42 slice (KFold fold / seed 43-44) |
| E30 | `e30_is3_f3` | `skip_leak` | leak — didn't hold out the seed-42 slice (KFold fold / seed 43-44) |
| E30 | `e30_is3_f4` | `skip_leak` | leak — didn't hold out the seed-42 slice (KFold fold / seed 43-44) |
| E12 | `prune12` | `skip_no_serialize` | no serialize row & not worth guessing (superseded/specialist) |
| E28/E33 | `member_0_t019` | `skip_no_serialize` | no serialize row & not worth guessing (superseded/specialist) |
| E28/E33 | `member_1_t023` | `skip_no_serialize` | no serialize row & not worth guessing (superseded/specialist) |
| E28/E33 | `member_2_t043` | `skip_no_serialize` | no serialize row & not worth guessing (superseded/specialist) |
| misc | `ft_Qwen__Qwen3-Embedding-0.6B` | `skip_no_serialize` | no serialize row & not worth guessing (superseded/specialist) |
| misc | `granite_tapt_mlm` | `skip_no_serialize` | no serialize row & not worth guessing (superseded/specialist) |
| misc | `granite_tapt_mlm_richargs` | `skip_no_serialize` | no serialize row & not worth guessing (superseded/specialist) |
| misc | `hist0` | `skip_no_serialize` | no serialize row & not worth guessing (superseded/specialist) |
| misc | `rdrop` | `skip_no_serialize` | no serialize row & not worth guessing (superseded/specialist) |
| misc | `v2_1024_full` | `skip_no_serialize` | no serialize row & not worth guessing (superseded/specialist) |
