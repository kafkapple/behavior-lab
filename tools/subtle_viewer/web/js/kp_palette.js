// AVATAR 11 body-part colors from BehaviorSplatter/configs/skeletons/avatar_11.yaml.
// The color is an anatomical display cue, not confidence, speed, or a behavior label.
import {color} from './colors.js';

const response = await fetch('/web/keypoint_palette.json');
if (!response.ok) throw new Error('Keypoint palette could not be loaded');
const AVATAR = await response.json();
const KEY = 'subtle.kpPalette';
export const PALETTES = {body: 'BS body parts', cluster: 'Legacy cluster edges'};
export let palette = Object.hasOwn(PALETTES, localStorage.getItem(KEY)) ? localStorage.getItem(KEY) : 'body';
export function setPalette(value) {
  if (!Object.hasOwn(PALETTES, value)) throw new Error(`unknown keypoint palette ${value}`);
  palette = value; localStorage.setItem(KEY, value);
}
export const partOf = (nodes, i) => AVATAR[nodes?.[i]] || null;
export const pointColor = (nodes, i, clusterId, is3d = false) =>
  palette === 'body' ? (partOf(nodes, i)?.[1] || '#7ee6b1') : is3d ? color(clusterId) : '#7ee6b1';
export const edgeColor = (nodes, target, clusterId) => palette === 'body' ? pointColor(nodes, target, clusterId) : color(clusterId);
