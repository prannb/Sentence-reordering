input_lists = [['dear', 'local', 'newspaper', 'think', 'effects', 'computers', 'people', 'great', 'learning', 'skillsaffects', 'give', 'us', 'time', 'chat', 'friendsnew', 'people', 'helps', 'us', 'learn', 'globe', 'astronomy', 'keeps', 'us', 'troble'], ['thing'], ['dont', 'think'], ['would', 'feel', 'teenager', 'always', 'phone', 'friends'], ['ever', 'time', 'chat', 'friends', 'buisness', 'partner', 'things'], ['well', 'new', 'way', 'chat', 'computer', 'plenty', 'sites', 'internet', 'facebook', 'myspace', 'ect'], ['think', 'setting', 'meeting', 'boss', 'computer', 'teenager', 'fun', 'phone', 'rushing', 'get', 'cause', 'want', 'use'], ['learn', 'countrysstates', 'outside'], ['well', 'computerinternet', 'new', 'way', 'learn', 'going', 'time'], ['might', 'think', 'child', 'spends', 'lot', 'time', 'computer', 'ask', 'question', 'economy', 'sea', 'floor', 'spreading', 'even', 'surprise', 'much', 'heshe', 'knows'], ['believe', 'computer', 'much', 'interesting', 'class', 'day', 'reading', 'books'], ['child', 'home', 'computer', 'local', 'library', 'better', 'friends', 'fresh', 'perpressured', 'something', 'know', 'isnt', 'right'], ['might', 'know', 'child', 'forbidde', 'hospital', 'bed', 'driveby'], ['rather', 'child', 'computer', 'learning', 'chatting', 'playing', 'games', 'safe', 'sound', 'home', 'community', 'place'], ['hope', 'reached', 'point', 'understand', 'agree', 'computers', 'great', 'effects', 'child', 'gives', 'us', 'time', 'chat', 'friendsnew', 'people', 'helps', 'us', 'learn', 'globe', 'believe', 'keeps', 'us', 'troble'], ['thank', 'listening']]

def m2n(input_lists = input_lists):
  output_lists = []
  num2word = {}
  word2num = {}

  count = 0

  for list in input_lists:
    output_lists.append([])
    for word in list:
      if word not in word2num:
        word2num[word] = count
        num2word[count] = word
        count += 1
      output_lists[-1].append(word2num[word])

  return output_lists, num2word, word2num
