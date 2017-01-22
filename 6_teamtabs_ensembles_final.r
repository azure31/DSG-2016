####-----------Create final ensembles---------------------########
####-----------author: TeamTabs, IIMC---------------------########

setwd('')

#Read in test data predictions from code files 4 & 5

vgg_192 = read.csv('vgg_192_tdg.csv')
vgg_224 = read.csv('vgg_224_tdg.csv')
vgg_256 = read.csv('vgg_256_tdg.csv')
gnet_192 = read.csv('gnet_192_tdg.csv')
gnet_224 = read.csv('gnet_224_tdg.csv')
gnet_256 = read.csv('gnet_256_tdg.csv')

vgg_192 = vgg_192[,-1]
vgg_224 = vgg_224[,-1]
vgg_256 = vgg_256[,-1]
gnet_192 = gnet_192[,-1]
gnet_224 = gnet_224[,-1]
gnet_256 = gnet_256[,-1]

colnames(vgg_256) = c("pred", "prob1", "prob2", "prob3", "prob4")
colnames(vgg_192) = c("pred", "prob1", "prob2", "prob3", "prob4")
colnames(vgg_224) = c("pred", "prob1", "prob2", "prob3", "prob4")
colnames(gnet_256) = c("pred", "prob1", "prob2", "prob3", "prob4")
colnames(gnet_224) = c("pred", "prob1", "prob2", "prob3", "prob4")
colnames(gnet_192) = c("pred", "prob1", "prob2", "prob3", "prob4")

id = read.csv('sample_submission4.csv')
final = vgg_256
for(i in 1:length(final$pred)){
  final$prob1[i] = mean(vgg_256$prob1[i], vgg_224$prob1[i], vgg_192$prob1[i], gnet_256$prob1[i], gnet_224$prob1[i], gnet_192$prob1[i])
  final$prob2[i] = mean(vgg_256$prob2[i], vgg_224$prob2[i], vgg_192$prob2[i], gnet_256$prob2[i], gnet_224$prob2[i], gnet_192$prob2[i])
  final$prob3[i] = mean(vgg_256$prob3[i], vgg_224$prob3[i], vgg_192$prob3[i], gnet_256$prob3[i], gnet_224$prob3[i], gnet_192$prob3[i])
  final$prob4[i] = mean(vgg_256$prob4[i], vgg_224$prob4[i], vgg_192$prob4[i], gnet_256$prob4[i], gnet_224$prob4[i], gnet_192$prob4[i])
  
  final$max[i] = max(final$prob1[i], final$prob2[i], final$prob3[i], final$prob4[i])
  final$pred[i] = ifelse(final$prob1[i]==final$max[i],1,ifelse(final$prob2[i]==final$max[i],2,ifelse(final$prob3[i]==final$max[i],3,4)))
  
  id$label[i] = final$pred[i]
  
  print(i)
}

write.csv('submission.csv', id)




