<script lang="ts">
  import { onMount } from 'svelte';
  // @ts-ignore
  import * as ONNX_WEBGPU from 'onnxruntime-web/webgpu';
  import { currentStatus, encoderOutput, inputImageData, fetchModel, modelSize } from 'src/lib';
  import {
    drawContour,
    drawImage,
    drawMask,
    prepareDecodingInputs,
    scaleAndProcessMasks,
  } from './utils';

  const ORIGINAL_SIZE = 1024;
  let canvas: HTMLCanvasElement;
  let maskThreshold = 1.5;
  let canvasSize: number;
  let scale: number;
  let offset: { x: number; y: number };
  let resizeObserver: ResizeObserver;
  let isClickDisabled = true;

  function updateImage(image: HTMLImageElement | ImageBitmap) {
    const context = canvas.getContext('2d');
    if (!context) return;
    const imageAspect = $inputImageData.width / $inputImageData.height;
    const canvasAspect = canvas.width / canvas.height;

    let drawWidth, drawHeight;
    let offsetX = 0, offsetY = 0;

    if (imageAspect > canvasAspect) {
      drawWidth = canvas.width;
      drawHeight = canvas.width / imageAspect;
      offsetY = (canvas.height - drawHeight) / 2;
    } else {
      drawHeight = canvas.height;
      drawWidth = canvas.height * imageAspect;
      offsetX = (canvas.width - drawWidth) / 2;
    }

    context.clearRect(0, 0, canvas.width, canvas.height);
    context.drawImage(image, offsetX, offsetY, drawWidth, drawHeight);

    canvasSize = Math.min(drawWidth, drawHeight);
    scale = canvasSize / ORIGINAL_SIZE;
    offset = { x: offsetX, y: offsetY };
  }

  $: if (canvas && $inputImageData) {
    createImageBitmap($inputImageData).then((bitmap) => {
      imageBitmap = bitmap;
      updateImage(bitmap);
    });
    if (!resizeObserver) {
      resizeObserver = new ResizeObserver(() => {
        if (canvas.parentElement) {
          canvas.width = canvas.parentElement.clientWidth;
          canvas.height = canvas.parentElement.clientHeight;
        }
      });
      resizeObserver.observe(canvas.parentElement);
    }
  }

  $: if ($encoderOutput) {
    isClickDisabled = false;
  }

  async function handleClick(event: MouseEvent) {
    if (isClickDisabled || !canvas || !$inputImageData || !$encoderOutput) return;

    const rect = canvas.getBoundingClientRect();
    const x = (event.clientX - rect.left - offset.x) / scale;
    const y = (event.clientY - rect.top - offset.y) / scale;

    console.log('Clicked position:', x, y);
    currentStatus.set(
      `Clicked on (${x}, ${y}). Downloading the decoder model if needed and generating masks...`,
    );

    const context = canvas.getContext('2d');
    if (!context || !imageBitmap) return;

    updateImage(imageBitmap);

    context.fillStyle = 'rgba(0, 0, 139, 0.7)'; // Dark blue with some transparency
    context.fillRect(x * scale + offset.x - 1, y * scale + offset.y - 1, 2, 2); // Smaller 2x2 pixel

    const inputPointCoords = new Float32Array([x, y, 0, 0]);
    const inputPointLabels = new Float32Array([1, -1]);
    const pointCoords = new ONNX_WEBGPU.Tensor(inputPointCoords, [1, 2, 2]);
    const pointLabels = new ONNX_WEBGPU.Tensor(inputPointLabels, [1, 2]);

    try {
      const decoderModel = await fetchModel({ isEncoder: false, modelSize: $modelSize });
      const decodingSession = await ONNX_WEBGPU.InferenceSession.create(decoderModel, {
        executionProviders: ['webgpu'],
      });
      // @ts-ignore
      const decodingFeeds = prepareDecodingInputs($encoderOutput, pointCoords, pointLabels);
      console.log({ decodingFeeds });
      const start = Date.now();
      const results = await decodingSession.run(decodingFeeds);
      const { masks, iou_predictions } = results;
      const stop = Date.now();
      const time_taken = (stop - start) / 1000;
      currentStatus.set(`Inference completed in ${time_taken} seconds`);

      const postProcessedMasks = scaleAndProcessMasks(masks, maskThreshold / 10);

      const bestMaskIndex = iou_predictions.data.indexOf(
        Math.max(...iou_predictions.data),
      );
      drawMask(context, 
        postProcessedMasks[bestMaskIndex],
        [255, 0, 0],
        0.5,
        ORIGINAL_SIZE,
        ORIGINAL_SIZE,
        canvasSize,
        offset,
      );

      drawContour(
        context,
        postProcessedMasks[bestMaskIndex],
        canvasSize,
        canvasSize,
        offset,
      );
    } catch (error) {
      console.error(error);
      currentStatus.set(`Error running inference: ${error}`);
    }
  }

  onMount(() => {
    return () => {
      if (resizeObserver) {
        resizeObserver.disconnect();
      }
    };
  });
</script>

{#if !$inputImageData}
  <div class="container"></div>
{:else}
  <div class="container-loaded">
    <div class="threshold-element">
      <label class="threshold-label" for="threshold">Mask&nbsp;Threshold:</label>
      <input type="range" id="threshold" min="0" max="20" step="0.1" bind:value={maskThreshold} />
      <span>{maskThreshold}</span>
    </div> 
    <canvas
      bind:this={canvas}
      on:click={handleClick}
      on:mouseover={() => {
        if (isClickDisabled) {
          currentStatus.set('Clicking is disabled until encoder output is available.');
        }
      }}
      on:focus={() => {
        if (isClickDisabled) {
          currentStatus.set('Clicking is disabled until encoder output is available.');
        }
      }}
      style:cursor={isClickDisabled ? 'not-allowed' : 'pointer'}
    />
  </div>
{/if}

<style>
  .container {
    display: flex;
    flex-direction: column;
    align-items: center;
    width: 100%;
    margin-bottom: 40px;
    gap: 8px;
  }
  .container-loaded {
    height: 60vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    width: 100%;
    margin-bottom: 40px;
    gap: 8px;
  }
  .threshold-element {
    width: 100%;
    display: flex;
    flex-direction: row;
    gap: 8px;
    justify-content: left;
    margin-bottom: 12px;
  }

  canvas {
    width: 100%;
    height: 100%;
  }

</style>
