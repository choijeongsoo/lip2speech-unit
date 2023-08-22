<meta name="viewport" content="width=device-width, initial-scale=1.0">

<h1>Intelligible Lip-to-Speech Synthesis with Speech Units</h1>

<h3>Jeongsoo Choi, Minsu Kim, Yong Man Ro</h3>

<h3><a href="https://arxiv.org/abs/2305.19603">[Paper]</a>  <a href="https://github.com/choijeongsoo/lip2speech-unit">[Code]</a></h3>

<div style="width:100%; margin-top: 20px; margin-bottom: 20px;">
<img align="center" src="imgs/fig1.png" style="  display: block;
  margin-left: auto;
  margin-right: auto;
  width: 100%;"  /></div>

<h2>Abstract</h2>
<div style="width:100%; max-width: 720px; margin-bottom: 20px; margin-left: auto; margin-right: auto;">
<p>In this paper, we propose a novel Lip-to-Speech synthesis (L2S) framework, for synthesizing intelligible speech from a silent lip movement video. Specifically, to complement the insufficient supervisory signal of the previous L2S model, we propose to use quantized self-supervised speech representations, named speech units, as an additional prediction target for the L2S model. Therefore, the proposed L2S model is trained to generate multiple targets, mel-spectrogram and speech units. As the speech units are discrete while mel-spectrogram is continuous, the proposed multi-target L2S model can be trained with strong content supervision, without using text-labeled data. Moreover, to accurately convert the synthesized mel-spectrogram into a waveform, we introduce a multi-input vocoder that can generate a clear waveform even from blurry and noisy mel-spectrogram by referring to the speech units. Extensive experimental results confirm the effectiveness of the proposed method in L2S.</p>
</div>

<h2>Random samples from LRS3 Dataset</h2>
<table border="0" width="100%">
  <thead>
    <tr>
      <th align="center"><strong>Silent video</strong></th>
      <th align="center"><strong>Ground Truth</strong></th>
      <th align="center"><strong>Ours<br>(Proposed + AV-HuBERT)</strong></th>
      <th align="center"><strong>Ours<br>(Proposed)</strong></th>
      <th align="center"><strong>Ours<br>(Proposed w/o aug)</strong></th>
      <th align="center"><strong>Multi-Task</strong></th>
      <th align="center"><strong>SVTS</strong></th>
      <th align="center"><strong>VCA-GAN</strong></th>
      <th align="center"><strong>Text</strong></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><video width="224" controls><source src="videos/silent/0gks6ceq4eQ_00006.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/gt/0gks6ceq4eQ_00006.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/proposed_avhubert/0gks6ceq4eQ_00006.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/proposed/0gks6ceq4eQ_00006.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/proposed_wo_aug/0gks6ceq4eQ_00006.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/multitask/0gks6ceq4eQ_00006.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/svts/0gks6ceq4eQ_00006.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/vcagan/0gks6ceq4eQ_00006.webm" type="video/webm"></video></td>
      <td wrap width="224">they are the basis of every action that you take</td>
<!--       <td wrap width="224">THEY ARE THE BASIS OF EVERY ACTION THAT YOU TAKE</td> -->
    </tr>
    <tr>
      <td><video width="224" controls><source src="videos/silent/7OMLSs8t1ng_00017.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/gt/7OMLSs8t1ng_00017.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/proposed_avhubert/7OMLSs8t1ng_00017.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/proposed/7OMLSs8t1ng_00017.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/proposed_wo_aug/7OMLSs8t1ng_00017.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/multitask/7OMLSs8t1ng_00017.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/svts/7OMLSs8t1ng_00017.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/vcagan/7OMLSs8t1ng_00017.webm" type="video/webm"></video></td>
      <td wrap width="224"><p>we don't trust the man</p></td>
<!--       <td wrap width="224"><p>WE DON'T TRUST THE MAN</p></td> -->
    </tr>
    <tr>
      <td><video width="224" controls><source src="videos/silent/8UhqkX2VAmo_00001.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/gt/8UhqkX2VAmo_00001.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/proposed_avhubert/8UhqkX2VAmo_00001.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/proposed/8UhqkX2VAmo_00001.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/proposed_wo_aug/8UhqkX2VAmo_00001.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/multitask/8UhqkX2VAmo_00001.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/svts/8UhqkX2VAmo_00001.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/vcagan/8UhqkX2VAmo_00001.webm" type="video/webm"></video></td>
      <td wrap width="224">otherwise millions more will die</td>
<!--       <td wrap width="224">OTHERWISE MILLIONS MORE WILL DIE</td> -->
    </tr>
    <tr>
      <td><video width="224" controls><source src="videos/silent/9uOMectkCCs_00002.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/gt/9uOMectkCCs_00002.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/proposed_avhubert/9uOMectkCCs_00002.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/proposed/9uOMectkCCs_00002.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/proposed_wo_aug/9uOMectkCCs_00002.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/multitask/9uOMectkCCs_00002.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/svts/9uOMectkCCs_00002.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/vcagan/9uOMectkCCs_00002.webm" type="video/webm"></video></td>
      <td wrap width="224">and then something falls off the wall</td>
<!--       <td wrap width="224">AND THEN SOMETHING FALLS OFF THE WALL</td> -->
    </tr>
    <tr>
      <td><video width="224" controls><source src="videos/silent/eVFYhbHpfqU_00007.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/gt/eVFYhbHpfqU_00007.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/proposed_avhubert/eVFYhbHpfqU_00007.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/proposed/eVFYhbHpfqU_00007.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/proposed_wo_aug/eVFYhbHpfqU_00007.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/multitask/eVFYhbHpfqU_00007.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/svts/eVFYhbHpfqU_00007.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/vcagan/eVFYhbHpfqU_00007.webm" type="video/webm"></video></td>
      <td wrap width="224">and that's powerful</td>
<!--       <td wrap width="224">AND THAT'S POWERFUL</td> -->
    </tr>
    <tr>
      <td><video width="224" controls><source src="videos/silent/gVfgkFaswn4_00002.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/gt/gVfgkFaswn4_00002.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/proposed_avhubert/gVfgkFaswn4_00002.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/proposed/gVfgkFaswn4_00002.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/proposed_wo_aug/gVfgkFaswn4_00002.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/multitask/gVfgkFaswn4_00002.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/svts/gVfgkFaswn4_00002.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/vcagan/gVfgkFaswn4_00002.webm" type="video/webm"></video></td>
      <td wrap width="224">so the answer to the second question can we change</td>
<!--       <td wrap width="224">SO THE ANSWER TO THE SECOND QUESTION CAN WE CHANGE</td> -->
    </tr>
    <tr>
      <td><video width="224" controls><source src="videos/silent/jbHpF7EySck_00001.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/gt/jbHpF7EySck_00001.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/proposed_avhubert/jbHpF7EySck_00001.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/proposed/jbHpF7EySck_00001.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/proposed_wo_aug/jbHpF7EySck_00001.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/multitask/jbHpF7EySck_00001.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/svts/jbHpF7EySck_00001.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/vcagan/jbHpF7EySck_00001.webm" type="video/webm"></video></td>
      <td wrap width="224">they want to be part of it</td>
<!--       <td wrap width="224">THEY WANT TO BE PART OF IT</td> -->
    </tr>
    <tr>
      <td><video width="224" controls><source src="videos/silent/jcp5vvxtEaU_00002.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/gt/jcp5vvxtEaU_00002.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/proposed_avhubert/jcp5vvxtEaU_00002.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/proposed/jcp5vvxtEaU_00002.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/proposed_wo_aug/jcp5vvxtEaU_00002.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/multitask/jcp5vvxtEaU_00002.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/svts/jcp5vvxtEaU_00002.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/vcagan/jcp5vvxtEaU_00002.webm" type="video/webm"></video></td>
      <td wrap width="224">we paid two to three times more than anybody else</td>
<!--       <td wrap width="224">WE PAID TWO TO THREE TIMES MORE THAN ANYBODY ELSE</td> -->
    </tr>
    <tr>
      <td><video width="224" controls><source src="videos/silent/NA7krbsdXFA_00011.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/gt/NA7krbsdXFA_00011.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/proposed_avhubert/NA7krbsdXFA_00011.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/proposed/NA7krbsdXFA_00011.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/proposed_wo_aug/NA7krbsdXFA_00011.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/multitask/NA7krbsdXFA_00011.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/svts/NA7krbsdXFA_00011.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/vcagan/NA7krbsdXFA_00011.webm" type="video/webm"></video></td>
      <td wrap width="224">and that's why it's been a pleasure speaking to you</td>
<!--       <td wrap width="224">AND THAT'S WHY IT'S BEEN A PLEASURE SPEAKING TO YOU</td> -->
    </tr>
    <tr>
      <td><video width="224" controls><source src="videos/silent/ni4FV5zL6lM_00001.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/gt/ni4FV5zL6lM_00001.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/proposed_avhubert/ni4FV5zL6lM_00001.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/proposed/ni4FV5zL6lM_00001.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/proposed_wo_aug/ni4FV5zL6lM_00001.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/multitask/ni4FV5zL6lM_00001.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/svts/ni4FV5zL6lM_00001.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/vcagan/ni4FV5zL6lM_00001.webm" type="video/webm"></video></td>
      <td wrap width="224">government officials are extremely mad</td>
<!--       <td wrap width="224">GOVERNMENT OFFICIALS ARE EXTREMELY MAD</td> -->
    </tr>
    <tr>
      <td><video width="224" controls><source src="videos/silent/O0JDDyqtSVY_00002.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/gt/O0JDDyqtSVY_00002.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/proposed_avhubert/O0JDDyqtSVY_00002.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/proposed/O0JDDyqtSVY_00002.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/proposed_wo_aug/O0JDDyqtSVY_00002.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/multitask/O0JDDyqtSVY_00002.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/svts/O0JDDyqtSVY_00002.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/vcagan/O0JDDyqtSVY_00002.webm" type="video/webm"></video></td>
      <td wrap width="224">but you know what</td>
<!--       <td wrap width="224">BUT YOU KNOW WHAT</td> -->
    </tr>
    <tr>
      <td><video width="224" controls><source src="videos/silent/ooAIIeo4AJQ_00003.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/gt/ooAIIeo4AJQ_00003.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/proposed_avhubert/ooAIIeo4AJQ_00003.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/proposed/ooAIIeo4AJQ_00003.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/proposed_wo_aug/ooAIIeo4AJQ_00003.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/multitask/ooAIIeo4AJQ_00003.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/svts/ooAIIeo4AJQ_00003.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/vcagan/ooAIIeo4AJQ_00003.webm" type="video/webm"></video></td>
      <td wrap width="224">thank you very much</td>
<!--       <td wrap width="224">THANK YOU VERY MUCH</td> -->
    </tr>
    <tr>
      <td><video width="224" controls><source src="videos/silent/QleRgTBMX88_00002.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/gt/QleRgTBMX88_00002.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/proposed_avhubert/QleRgTBMX88_00002.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/proposed/QleRgTBMX88_00002.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/proposed_wo_aug/QleRgTBMX88_00002.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/multitask/QleRgTBMX88_00002.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/svts/QleRgTBMX88_00002.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/vcagan/QleRgTBMX88_00002.webm" type="video/webm"></video></td>
      <td wrap width="224">beth israel's in boston</td>
<!--       <td wrap width="224">BETH ISRAEL'S IN BOSTON</td> -->
    </tr>
    <tr>
      <td><video width="224" controls><source src="videos/silent/Sa27SUR0Mlo_00003.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/gt/Sa27SUR0Mlo_00003.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/proposed_avhubert/Sa27SUR0Mlo_00003.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/proposed/Sa27SUR0Mlo_00003.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/proposed_wo_aug/Sa27SUR0Mlo_00003.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/multitask/Sa27SUR0Mlo_00003.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/svts/Sa27SUR0Mlo_00003.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/vcagan/Sa27SUR0Mlo_00003.webm" type="video/webm"></video></td>
      <td wrap width="224">but we're not there yet</td>
<!--       <td wrap width="224">BUT WE'RE NOT THERE YET</td> -->
    </tr>
    <tr>
      <td><video width="224" controls><source src="videos/silent/Sv5QitqbxJw_00004.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/gt/Sv5QitqbxJw_00004.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/proposed_avhubert/Sv5QitqbxJw_00004.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/proposed/Sv5QitqbxJw_00004.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/proposed_wo_aug/Sv5QitqbxJw_00004.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/multitask/Sv5QitqbxJw_00004.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/svts/Sv5QitqbxJw_00004.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/vcagan/Sv5QitqbxJw_00004.webm" type="video/webm"></video></td>
      <td wrap width="224">and they don't need to ask for permission</td>
<!--       <td wrap width="224">AND THEY DON'T NEED TO ASK FOR PERMISSION</td> -->
    </tr>
    <tr>
      <td><video width="224" controls><source src="videos/silent/TUxwiVFgghE_00001.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/gt/TUxwiVFgghE_00001.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/proposed_avhubert/TUxwiVFgghE_00001.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/proposed/TUxwiVFgghE_00001.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/proposed_wo_aug/TUxwiVFgghE_00001.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/multitask/TUxwiVFgghE_00001.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/svts/TUxwiVFgghE_00001.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/vcagan/TUxwiVFgghE_00001.webm" type="video/webm"></video></td>
      <td wrap width="224">how much they do</td>
<!--       <td wrap width="224">HOW MUCH THEY DO</td> -->
    </tr>
    <tr>
      <td><video width="224" controls><source src="videos/silent/UZmXwOgNq7c_00002.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/gt/UZmXwOgNq7c_00002.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/proposed_avhubert/UZmXwOgNq7c_00002.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/proposed/UZmXwOgNq7c_00002.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/proposed_wo_aug/UZmXwOgNq7c_00002.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/multitask/UZmXwOgNq7c_00002.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/svts/UZmXwOgNq7c_00002.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/vcagan/UZmXwOgNq7c_00002.webm" type="video/webm"></video></td>
      <td wrap width="224">so what can we do</td>
<!--       <td wrap width="224">SO WHAT CAN WE DO</td> -->
    </tr>
    <tr>
      <td><video width="224" controls><source src="videos/silent/wlR1ojoiue0_00001.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/gt/wlR1ojoiue0_00001.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/proposed_avhubert/wlR1ojoiue0_00001.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/proposed/wlR1ojoiue0_00001.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/proposed_wo_aug/wlR1ojoiue0_00001.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/multitask/wlR1ojoiue0_00001.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/svts/wlR1ojoiue0_00001.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/vcagan/wlR1ojoiue0_00001.webm" type="video/webm"></video></td>
      <td wrap width="224">they were wonderful people</td>
<!--       <td wrap width="224">THEY WERE WONDERFUL PEOPLE</td> -->
    </tr>
    <tr>
      <td><video width="224" controls><source src="videos/silent/Zo62S0ulqhA_00001.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/gt/Zo62S0ulqhA_00001.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/proposed_avhubert/Zo62S0ulqhA_00001.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/proposed/Zo62S0ulqhA_00001.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/proposed_wo_aug/Zo62S0ulqhA_00001.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/multitask/Zo62S0ulqhA_00001.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/svts/Zo62S0ulqhA_00001.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/vcagan/Zo62S0ulqhA_00001.webm" type="video/webm"></video></td>
      <td wrap width="224">how do you change your behavior</td>
<!--       <td wrap width="224">HOW DO YOU CHANGE YOUR BEHAVIOR</td> -->
    </tr>
    <tr>
      <td><video width="224" controls><source src="videos/silent/Zo62S0ulqhA_00003.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/gt/Zo62S0ulqhA_00003.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/proposed_avhubert/Zo62S0ulqhA_00003.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/proposed/Zo62S0ulqhA_00003.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/proposed_wo_aug/Zo62S0ulqhA_00003.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/multitask/Zo62S0ulqhA_00003.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/svts/Zo62S0ulqhA_00003.webm" type="video/webm"></video></td>
      <td><video width="224" controls><source src="videos/vcagan/Zo62S0ulqhA_00003.webm" type="video/webm"></video></td>
      <td wrap width="224">he could come over and help me</td>
<!--       <td wrap width="224">HE COULD COME OVER AND HELP ME</td> -->
    </tr>
  
  </tbody>
</table>
