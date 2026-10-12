'use strict';
// 実験用の薄いラッパ。探索の本体は public/search.js(Web UI と共通)。
const NN = require('./public/nn');
const { makeSearch } = require('./public/search');

function makeSearchPolicy(net, depth) {
  const choose = makeSearch(net, depth);
  return (board, player, mask) => choose(board, player, mask).move;
}

module.exports = { makeSearchPolicy, makeSearch, loadNet: (json) => NN.fromJSON(json) };
