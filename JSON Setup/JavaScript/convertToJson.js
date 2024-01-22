const sip_dictionary = require('./data');

const jsonContent = JSON.stringify(sip_dictionary, null, 4);
console.log(jsonContent);

// Opcional: Salvar em um arquivo
const fs = require('fs');
fs.writeFileSync('output.json', jsonContent);

