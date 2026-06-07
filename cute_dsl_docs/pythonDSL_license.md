

Software License Agreement — NVIDIA CUTLASS Documentation



[Skip to main content](#main-content)

Back to top




`Ctrl`+`K`

[![NVIDIA CUTLASS Documentation - Home](../../../_static/nvidia-logo-horiz-rgb-blk-for-screen.svg)
![NVIDIA CUTLASS Documentation - Home](../../../_static/nvidia-logo-horiz-rgb-wht-for-screen.svg)

NVIDIA CUTLASS Documentation](../../../index.html)

Choose version

Search
`Ctrl`+`K`

Search
`Ctrl`+`K`

[![NVIDIA CUTLASS Documentation - Home](../../../_static/nvidia-logo-horiz-rgb-blk-for-screen.svg)
![NVIDIA CUTLASS Documentation - Home](../../../_static/nvidia-logo-horiz-rgb-wht-for-screen.svg)

NVIDIA CUTLASS Documentation](../../../index.html)

Choose version

Table of Contents

* [Changelog](../../../CHANGELOG.html)

CuTe DSL

* [Overview](overview.html)
* [Functionality](functionality.html)
* [Quick Start Guide](quick_start.html)
* [CuTe DSL](cute_dsl.html)
  + [Introduction](cute_dsl_general/dsl_introduction.html)
  + [Code Generation](cute_dsl_general/dsl_code_generation.html)
  + [Control Flow](cute_dsl_general/dsl_control_flow.html)
  + [JIT Argument Generation](cute_dsl_general/dsl_jit_arg_generation.html)
  + [JIT Argument: Layouts](cute_dsl_general/dsl_dynamic_layout.html)
  + [JIT Caching](cute_dsl_general/dsl_jit_caching.html)
  + [JIT Compilation Options](cute_dsl_general/dsl_jit_compilation_options.html)
  + [JIT Types](cute_dsl_general/types.html)
  + [Integration with Frameworks](cute_dsl_general/framework_integration.html)
  + [Debugging with the DSL](cute_dsl_general/debugging.html)
  + [Autotuning with the DSL](cute_dsl_general/autotuning_gemm.html)
  + [Educational Notebooks](cute_dsl_general/notebooks.html)
  + [Deprecation Policy](deprecation.html)
  + [Compile with TVM FFI](cute_dsl_general/compile_with_tvm_ffi.html)
  + [Ahead-of-Time (AOT) Compilation](cute_dsl_general/dsl_ahead_of_time_compilation.html)
  + [Talks and Presentations](cute_dsl_general/resources.html)
* [CuTe DSL API](cute_dsl_api.html)
  + [changelog](cute_dsl_api/changelog.html)
  + [cute](cute_dsl_api/cute.html)
    - [arch](cute_dsl_api/cute_arch.html)
    - [Runtime](cute_dsl_api/cute_runtime.html)
  + [cute\_nvgpu](cute_dsl_api/cute_nvgpu.html)
    - [Common](cute_dsl_api/cute_nvgpu_common.html)
    - [warp submodule](cute_dsl_api/cute_nvgpu_warp.html)
    - [warpgroup submodule](cute_dsl_api/cute_nvgpu_warpgroup.html)
    - [cpasync submodule](cute_dsl_api/cute_nvgpu_cpasync.html)
    - [tcgen05 submodule](cute_dsl_api/cute_nvgpu_tcgen05.html)
  + [pipeline](cute_dsl_api/pipeline.html)
  + [utils](cute_dsl_api/utils.html)
    - [Utilities for SM90](cute_dsl_api/utils_sm90.html)
    - [Utilities for SM100](cute_dsl_api/utils_sm100.html)
* [Limitations](limitations.html)
* [FAQs](faqs.html)

CUTLASS C++

* [Overview](../../../overview.html)
* [Getting Started](../cpp/getting_started.html)
  + [Quickstart](../cpp/quickstart.html)
  + [IDE Setup](../cpp/ide_setup.html)
  + [Build](../cpp/build/index.html)
    - [Building on Windows with Visual Studio](../cpp/build/building_in_windows_with_visual_studio.html)
    - [Building with Clang as host compiler](../cpp/build/building_with_clang_as_host_compiler.html)
  + [Functionality](../cpp/functionality.html)
  + [Terminology](../cpp/terminology.html)
  + [Fundamental Types](../cpp/fundamental_types.html)
  + [Programming Guidelines](../cpp/programming_guidelines.html)
  + [GEMM Heuristics](../cpp/heuristics.html)
* [Efficient GEMM in CUDA](../cpp/efficient_gemm.html)
* [Synchronization primitives](../cpp/pipeline.html)
* [CUTLASS Profiler](../cpp/profiler.html)
* [Dependent Kernel Launch](../cpp/dependent_kernel_launch.html)
* [Blackwell Specific](../cpp/blackwell.html)
  + [Blackwell SM100 GEMMs](../cpp/blackwell_functionality.html)
  + [Blackwell Cluster Launch Control](../cpp/blackwell_cluster_launch_control.html)
* [CuTe](../cpp/cute/index.html)
  + [00\_quickstart](../cpp/cute/00_quickstart.html)
  + [01\_layout](../cpp/cute/01_layout.html)
  + [02\_layout\_algebra](../cpp/cute/02_layout_algebra.html)
  + [03\_tensor](../cpp/cute/03_tensor.html)
  + [04\_algorithms](../cpp/cute/04_algorithms.html)
  + [0t\_mma\_atom](../cpp/cute/0t_mma_atom.html)
  + [0x\_gemm\_tutorial](../cpp/cute/0x_gemm_tutorial.html)
  + [0y\_predication](../cpp/cute/0y_predication.html)
  + [0z\_tma\_tensors](../cpp/cute/0z_tma_tensors.html)
* [CUTLASS 3.x](../cpp/cutlass_3x.html)
  + [Design](../cpp/cutlass_3x_design.html)
  + [GEMM Backwards Compatibility](../cpp/cutlass_3x_backwards_compatibility.html)
  + [GEMM API](../cpp/gemm_api_3x.html)
* [CUTLASS 2.x](../cpp/cutlass_2x.html)
  + [Layouts and Tensors](../cpp/layout.html)
  + [GEMM API](../cpp/gemm_api.html)
  + [Tile Iterator Concepts](../cpp/tile_iterator_concept.html)
  + [Utilities](../cpp/utilities.html)
* [Code Organization](../cpp/code_organization.html)
* [Grouped Kernel Schedulers](../cpp/grouped_scheduler.html)
* [CUTLASS Convolution](../cpp/implicit_gemm_convolution.html)

Reference

* [Software License Agreement](#)

* Software License Agreement

# Software License Agreement[#](#software-license-agreement "Link to this heading")

IMPORTANT NOTICE â PLEASE READ AND AGREE BEFORE USING THE SOFTWARE
This software license agreement (âAgreementâ) is a legal agreement between you, whether an individual or entity, (âyouâ) and NVIDIA Corporation (âNVIDIAâ) and governs the use of the NVIDIA CUTLASS DSLs software and materials that NVIDIA delivers to you under this Agreement (âSoftwareâ).
NVIDIA and you are each a âpartyâ and collectively the âparties.â
This Agreement can be accepted only by an adult of legal age of majority in the country in which the Software is used.
If you donât have the required age or authority to accept this Agreement, or if you donât accept all the terms and conditions of this Agreement, do not use the Software.

## 1. License Grants[#](#license-grants "Link to this heading")

1.1. License Grant to You. The Software made available by NVIDIA to you is licensed, not sold.
Subject to the terms of this Agreement, NVIDIA grants you a limited, non-exclusive, revocable, non-transferable, and non-sublicensable (except as expressly granted in this Agreement), license to:

1. install and use copies of the Software,
2. configure the Software using configuration files provided (if applicable),
3. modify and create derivative works of any sample or example source code NVIDIA delivers to you as part of the Software (âDerivativesâ) (if applicable), and
4. distribute python files in the Software package in source format as incorporated into a software application subject to the following distribution requirements:

   1. Your application must have material additional functionality, beyond the included portions of the Software.
   2. The distributable portions of the Software shall only be accessed by your application.
   3. The following notice shall be included in modifications and derivative works of sample source code distributed: âThis software contains source code provided by NVIDIA Corporation.â
   4. Unless a developer tool is identified in this Agreement as distributable, it is delivered for your internal use only.
   5. The terms under which you distribute your application must be consistent with the terms of this Agreement, including (without limitation) terms relating to the license grant and license restrictions and protection of NVIDIAâs intellectual property rights.
   6. Additionally, you agree that you will protect the privacy, security and legal rights of your application users.

The foregoing (a) through (d) are, collectively, the âPurposeâ, and the developed applications are only for use in systems with NVIDIA GPUs.

1.2. License Grant to NVIDIA. Subject to the terms of this Agreement, you grant NVIDIA and its affiliates a non-exclusive, perpetual, irrevocable, sublicensable, worldwide, royalty-free, fully paid-up and transferable license, under your intellectual property rights, to publicly perform, publicly display, reproduce, use, make, have made, sell, offer for sale, distribute (through multiple tiers of distribution), import, create derivative works of and otherwise commercialize and exploit at NVIDIAâs discretion any Derivatives created by or for you.
You may, but are not required to, deliver any Derivatives to NVIDIA.

## 2. License Restrictions[#](#license-restrictions "Link to this heading")

Your license to use the Software and Derivatives is restricted as stated in this Section 2 (âLicense Restrictionsâ).
You will cooperate with NVIDIA and, upon NVIDIAâs written request, you will confirm in writing and provide reasonably requested information to verify your compliance with the terms of this Agreement.
You may not:

2.1. Use the Software or Derivatives for any purpose other than the Purpose;

2.2. Sell, rent, sublicense, transfer, distribute or otherwise make available to others (except authorized users as stated in Section 3 (âAuthorized Usersâ)) any portion of the Software or Derivatives, except as expressly granted in Section 1.1 (âLicense Grant to Youâ);

2.3. Reverse engineer, decompile, or disassemble the Software components provided in binary form, nor attempt in any other manner to obtain source code of such Software;

2.4. Modify or create derivative works of the Software, except as expressly granted in Section 1.1 (âLicense Grant to Youâ);

2.5. Change or remove copyright or other proprietary notices in the Software;

2.6. Bypass, disable, or circumvent any technical limitation, encryption, security, digital rights management or authentication mechanism in the Software;

2.7. Use the Software or Derivatives in any manner that would cause them to become subject to an open source software license, subject to the terms in Section 6 (âComponents Under Other Licensesâ);

2.8. Use the Software or Derivatives in violation of any applicable law or regulation in relevant jurisdictions

2.9. Indicate that a product or service developed with the Software or Derivatives is sponsored or endorsed by NVIDIA;

2.10. Replace any NVIDIA software components in the Software that are governed by this Agreement with other software that implements NVIDIA APIs;

2.11. Reverse engineer, decompile or disassemble any portion of the output generated using Software elements for the purpose of translating such output artifacts to target a non-NVIDIA platform; or

## 3. Authorized Users[#](#authorized-users "Link to this heading")

You may allow employees and contractors of your entity or of your subsidiary(ies), and for educational institutions also enrolled students, to internally access and use the Software as authorized by this Agreement from your secure network to perform the work authorized by this Agreement on your behalf.
You are responsible for the compliance with the terms of this Agreement by your authorized users.
Any act or omission that if committed by you would constitute a breach of this Agreement will be deemed to constitute a breach of this Agreement if committed by your authorized users.

## 4. Pre-Release[#](#pre-release "Link to this heading")

Software versions identified as alpha, beta, preview, early access or otherwise as pre-release (âPre-Releaseâ) may not be fully functional, may contain errors or design flaws, and may have reduced or different security, privacy, availability and reliability standards relative to NVIDIA commercial offerings.
You use Pre-Release Software at your own risk. NVIDIA did not design or test the Software for use in production or business-critical systems.
NVIDIA may choose not to make available a commercial version of Pre-Release Software.
NVIDIA may also choose to abandon development and terminate the availability of Pre-Release Software at any time without liability.

## 5. Updates[#](#updates "Link to this heading")

NVIDIA may at any time and at its option, change, discontinue, or deprecate any part, or all, of the Software, or change or remove features or functionality, or make available patches, workarounds or other updates to the Software.
Unless the updates are provided with their separate governing terms, they are deemed part of the Software licensed to you under this Agreement, and your continued use of the Software is deemed acceptance of such changes.

## 6. Components Under Other Licenses[#](#components-under-other-licenses "Link to this heading")

The Software may include or be distributed with components provided with separate legal notices or terms that accompany the components, such as open source software licenses and other license terms (âOther Licensesâ).
The components are subject to the applicable Other Licenses, including any proprietary notices, disclaimers, requirements and extended use rights;
except that this Agreement will prevail regarding the use of third-party open source software, unless a third-party open source software license requires its license terms to prevail.
Open source software license means any software, data or documentation subject to any license identified as an open source license by the Open Source Initiative (<http://opensource.org>), Free Software Foundation (<http://www.fsf.org>) or other similar open source organization or listed by the Software Package Data Exchange (SPDX) Workgroup under the Linux Foundation (<http://www.spdx.org>).

## 7. Ownership[#](#ownership "Link to this heading")

7.1. NVIDIA Ownership. The Software, including all intellectual property rights, is and will remain the sole and exclusive property of NVIDIA or its licensors.
Except as expressly granted in this Agreement, (a) NVIDIA reserves all rights, interests and remedies in connection with the Software, and (b) no other license or right is granted to you by implication, estoppel or otherwise.

7.2. Your Ownership. Subject to the rights of NVIDIA and its suppliers in the Software, which continue to be licensed as stated in this Agreement, even when incorporated in your products or services, and the extent permitted by applicable law, as between you and NVIDIA, you hold all rights, title and interest in and to your products, services and Derivatives you develop as permitted in this Agreement including their respective intellectual property rights.

## 8. Feedback[#](#feedback "Link to this heading")

You may, but you are not obligated to, provide suggestions, requests, fixes, modifications, enhancements, or other feedback regarding the Software (collectively, âFeedbackâ).
Feedback, even if designated as confidential by you, will not create any confidentiality obligation for NVIDIA or its affiliates.
If you provide Feedback, you grant NVIDIA, its affiliates and its designees a non-exclusive, perpetual, irrevocable, sublicensable, worldwide, royalty-free, fully paid-up and transferable license, under your intellectual property rights, to publicly perform, publicly display, reproduce, use, make, have made, sell, offer for sale, distribute (through multiple tiers of distribution), import, create derivative works of and otherwise commercialize and exploit the Feedback at NVIDIAâs discretion.

## 9. Termination[#](#termination "Link to this heading")

9.1. Termination. This Agreement will automatically terminate without notice from NVIDIA if you fail to comply with any of the terms in this Agreement or if you commence or participate in any legal proceeding against NVIDIA with respect to the Software.
Additionally, either party may terminate this Agreement at any time with thirty (30) daysâ advance written notice to the other party.

9.2. Effect of Termination. Upon any expiration or termination of this Agreement, you will promptly (a) stop using and return, delete or destroy NVIDIA confidential information and all Software received under this Agreement, and (b) delete or destroy Derivatives created under this Agreement, unless an authorized NVIDIA representative provides prior written approval that you may keep a copy of the Derivatives solely for archival purposes.
Upon written request, you will certify in writing that you have complied with your obligations under this Section 9.2 (âEffect of Terminationâ).

9.3. Survival. Section 1.2 (âLicense Grant to NVIDIAâ), Section 5 (âUpdatesâ), Section 6 (âComponents Under Other Licensesâ), Section 7 (âOwnershipâ), Section 8 (âFeedback), Section 9.2 (âEffect of Terminationâ), Section 9.3 (âSurvivalâ), Section 10 (âDisclaimer of Warrantiesâ), Section 11 (âLimitation of Liabilityâ), Section 12 (âUse in Mission Critical Applicationsâ), Section 13 (âGoverning Law and Jurisdictionâ), Section 14 (âIndemnityâ) and Section 15 (âGeneralâ) will survive any expiration or termination of this Agreement.

## 10. Disclaimer of Warranties[#](#disclaimer-of-warranties "Link to this heading")

THE SOFTWARE IS PROVIDED BY NVIDIA AS-IS AND WITH ALL FAULTS. TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, NVIDIA DISCLAIMS ALL WARRANTIES AND REPRESENTATIONS OF ANY KIND, WHETHER
EXPRESS, IMPLIED OR STATUTORY, RELATING TO OR ARISING UNDER THIS AGREEMENT, INCLUDING, WITHOUT LIMITATION, THE WARRANTIES OF TITLE, NONINFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, USAGE OF TRADE AND COURSE OF DEALING. NVIDIA DOES NOT WARRANT OR ASSUME RESPONSIBILITY FOR THE ACCURACY OR COMPLETENESS OF ANY THIRD-PARTY INFORMATION, TEXT, GRAPHICS, LINKS CONTAINED IN THE SOFTWARE.
WITHOUT LIMITING THE FOREGOING, NVIDIA DOES NOT WARRANT THAT THE SOFTWARE WILL MEET YOUR REQUIREMENTS, ANY DEFECTS OR ERRORS WILL BE CORRECTED, ANY CERTAIN CONTENT WILL BE AVAILABLE; OR THAT THE SOFTWARE IS FREE OF VIRUSES OR OTHER HARMFUL COMPONENTS. NO INFORMATION OR ADVICE GIVEN BY NVIDIA WILL IN ANY WAY INCREASE THE SCOPE OF ANY WARRANTY EXPRESSLY PROVIDED IN THIS AGREEMENT.
NVIDIA does not warrant or assume responsibility for the accuracy or completeness of any third-party information, text, graphics or links contained in the Software.

## 11. Limitations of Liability[#](#limitations-of-liability "Link to this heading")

11.1. EXCLUSIONS. TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, IN NO EVENT WILL NVIDIA BE LIABLE FOR ANY (I) INDIRECT, PUNITIVE, SPECIAL, INCIDENTAL OR CONSEQUENTIAL DAMAGES, OR (ii) DAMAGES FOR (a) THE COST OF PROCURING SUBSTITUTE GOODS, OR (b) LOSS OF PROFITS, REVENUES, USE, DATA OR GOODWILL ARISING OUT OF OR RELATED TO THIS AGREEMENT, WHETHER BASED ON BREACH OF CONTRACT, TORT (INCLUDING NEGLIGENCE), STRICT LIABILITY, OR OTHERWISE, AND EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES AND EVEN IF A PARTYâS REMEDIES FAIL THEIR ESSENTIAL PURPOSE.

11.2. DAMAGES CAP. ADDITIONALLY, TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, NVIDIAâS TOTAL CUMULATIVE AGGREGATE LIABILITY FOR ANY AND ALL LIABILITIES, OBLIGATIONS OR CLAIMS ARISING OUT OF OR RELATED TO THIS AGREEMENT WILL NOT EXCEED FIVE U.S. DOLLARS (US$5).

## 12. Use in Mission Critical Applications[#](#use-in-mission-critical-applications "Link to this heading")

You acknowledge that the Software provided under this Agreement is not designed or tested by NVIDIA for use in any system or application where the use or failure of such system or application developed with NVIDIAâs Software could result in injury, death or catastrophic damage (each, a âMission Critical Applicationâ).
Examples of Mission Critical Applications include use in avionics, navigation, autonomous vehicle applications, AI solutions for automotive products, military, medical, life support or other mission-critical or life-critical applications.
NVIDIA will not be liable to you or any third party, in whole or in part, for any claims or damages arising from these uses.
You are solely responsible for ensuring that systems and applications developed with the Software include sufficient safety and redundancy features and comply with all applicable legal and regulatory standards and requirements.

## 13. Governing Law and Jurisdiction[#](#governing-law-and-jurisdiction "Link to this heading")

This Agreement will be governed in all respects by the laws of the United States and the laws of the State of Delaware, without regard to conflict of laws principles or the United Nations Convention on Contracts for the International Sale of Goods.
The state and federal courts residing in Santa Clara County, California will have exclusive jurisdiction over any dispute or claim arising out of or related to this Agreement, and the parties irrevocably consent to personal jurisdiction and venue in those courts;
except that either party may apply for injunctive remedies or an equivalent type of urgent legal relief in any jurisdiction.

## 14. Indemnity[#](#indemnity "Link to this heading")

By using the Software you agree to defend, indemnify and hold harmless NVIDIA and its affiliates and their respective officers, directors, employees and agents from and against any claims, disputes, demands, liabilities, damages, losses, costs and expenses arising out of or in any way connected with (i) products or services that have been developed or deployed with or use the Software, or claims that they violate laws, or infringe, violate, or misappropriate any third party right;
or (ii) use of the Software in breach of the terms of this Agreement.

## 15. General[#](#general "Link to this heading")

15.1. Independent Contractors.
:   The parties are independent contractors, and this Agreement does not create a joint venture, partnership, agency, or other form of business association between the parties.
    Neither party will have the power to bind the other party or incur any obligation on its behalf without the other partyâs prior written consent.
    Nothing in this Agreement prevents either party from participating in similar arrangements with third parties.

15.2. No Assignment.
:   NVIDIA may assign, delegate or transfer its rights or obligations under this Agreement by any means or operation of law.
    You may not, without NVIDIAâs prior written consent, assign, delegate or transfer any of your rights or obligations under this Agreement by any means or operation of law, and any attempt to do so is null and void.

15.3. No Waiver.
:   No failure or delay by a party to enforce any term or obligation of this Agreement will operate as a waiver by that party, or prevent the enforcement of such term or obligation later.

15.4. Trade Compliance.
:   You agree to comply with all applicable export, import, trade and economic sanctions laws and regulations, as amended, including without limitation U.S. Export Administration Regulations and Office of Foreign Assets Control regulations.
    You confirm (a) your understanding that export or reexport of certain NVIDIA products or technologies may require a license or other approval from appropriate authorities and (b) that you will not export or reexport any products or technology, directly or indirectly, without first obtaining any required license or other approval from appropriate authorities, (i) to any countries that are subject to any U.S. or local export restrictions (currently including, but not necessarily limited to, Belarus, Cuba, Iran, North Korea, Russia, Syria, the Region of Crimea, Donetsk Peopleâs Republic Region and Luhansk Peopleâs Republic Region);
    (ii) to any end-user who you know or have reason to know will utilize them in the design, development or production of nuclear, chemical or biological weapons, missiles, rocket systems, unmanned air vehicles capable of a maximum range of at least 300 kilometers, regardless of payload, or intended for military end-use, or any weapons of mass destruction;
    (iii) to any end-user who has been prohibited from participating in the U.S. or local export transactions by any governing authority;
    or (iv) to any known military or military-intelligence end-user or for any known military or military-intelligence end-use in accordance with U.S. trade compliance laws and regulations.

15.5. Government Rights.
:   The Software, documentation and technology (âProtected Itemsâ) are âCommercial productsâ as this term is defined at 48 C.F.R.
    2.101, consisting of âcommercial computer softwareâ and âcommercial computer software documentationâ as such terms are used in, respectively, 48 C.F.R.
    12.212 and 48 C.F.R. 227.7202 & 252.227-7014(a)(1). Before any Protected Items are supplied to the U.S. Government, you will (i) inform the U.S. Government in writing that the Protected Items are and must be treated as commercial computer software and commercial computer software documentation developed at private expense;
    (ii) inform the U.S. Government that the Protected Items are provided subject to the terms of the Agreement;
    and (iii) mark the Protected Items as commercial computer software and commercial computer software documentation developed at private expense.
    In no event will you permit the U.S. Government to acquire rights in Protected Items beyond those specified in 48 C.F.R.
    52.227-19(b)(1)-(2) or 252.227-7013(c) except as expressly approved by NVIDIA in writing.

15.6. Notices.
:   Please direct your legal notices or other correspondence to [legalnotices@nvidia.com](mailto:legalnotices%40nvidia.com) with a copy mailed to NVIDIA Corporation, 2788 San Tomas Expressway, Santa Clara, California 95051, United States of America, Attention: Legal Department.
    If NVIDIA needs to contact you, you consent to receive the notices by email and agree that such notices will satisfy any legal communication requirements.

15.7. Severability.
:   If a court of competent jurisdiction rules that a provision of this Agreement is unenforceable, that provision will be deemed modified to the extent necessary to make it enforceable and the remainder of this Agreement will continue in full force and effect.

15.8. Amendment.
:   Any amendment to this Agreement must be in writing and signed by authorized representatives of both parties.

15.9. Construction.
:   The headings in the Agreement are included solely for convenience and are not intended to affect the meaning or interpretation of the Agreement.
    As required by the context of the Agreement, the singular of a term includes the plural and vice versa.

15.10. Force Majeure.
:   Neither party will be liable during any period where an event or circumstance prevents or delays that party from performing its obligations under this Agreement and that event or circumstance: (i) is not within the reasonable control of that party and is not the result of that partyâs negligence, and (ii) cannot be overcome or avoided by that party using reasonably diligent efforts.

15.11. Entire Agreement.
:   Regarding the subject matter of this Agreement, the parties agree that (a) this Agreement constitutes the entire and exclusive agreement between the parties and supersedes all prior and contemporaneous communications and (b) any additional or different terms or conditions, whether contained in purchase orders, order acknowledgments, invoices or otherwise, will not be binding and are null and void.

(v. May 8, 2025)

[previous

CUTLASS Convolution](../cpp/implicit_gemm_convolution.html "previous page")

On this page

* [1. License Grants](#license-grants)
* [2. License Restrictions](#license-restrictions)
* [3. Authorized Users](#authorized-users)
* [4. Pre-Release](#pre-release)
* [5. Updates](#updates)
* [6. Components Under Other Licenses](#components-under-other-licenses)
* [7. Ownership](#ownership)
* [8. Feedback](#feedback)
* [9. Termination](#termination)
* [10. Disclaimer of Warranties](#disclaimer-of-warranties)
* [11. Limitations of Liability](#limitations-of-liability)
* [12. Use in Mission Critical Applications](#use-in-mission-critical-applications)
* [13. Governing Law and Jurisdiction](#governing-law-and-jurisdiction)
* [14. Indemnity](#indemnity)
* [15. General](#general)





[![NVIDIA](../../../_static/nvidia-logo-horiz-rgb-1c-blk-for-screen.svg)
![NVIDIA](../../../_static/nvidia-logo-horiz-rgb-1c-wht-for-screen.svg)](https://www.nvidia.com)

Copyright Â© 2025-2026, NVIDIA Corporation.

Last updated on Apr 08, 2026.