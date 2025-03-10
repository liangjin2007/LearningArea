## 修改USkeletalMesh的法向
```
FSkeletalMeshModel* TargetSkelMeshResource = TargetMesh->GetImportedModel();
FSkeletalMeshLODModel& TargetModel = TargetSkelMeshResource->LODModels[LODIndex];
FSkeletalMeshLODRenderData& TargetLODRenderData = TargetMesh->GetResourceForRendering()->LODRenderData[LODIndex];
FMeshDescription& TargetDescription = *TargetMesh->GetMeshDescription(LODIndex);
FSkeletalMeshAttributes TargetMeshAttributes(TargetDescription);
TVertexInstanceAttributesRef<FVector3f> TargetVertexInstanceNormals = TargetMeshAttributes.GetVertexInstanceNormals();
TVertexInstanceAttributesRef<FVector3f> TargetVertexInstanceTangents = TargetMeshAttributes.GetVertexInstanceTangents();
TargetModel.Sections[TargetSectionIndex].SoftVertices[TargetVertexIndexInSection].TangentZ = NewTangentZ;

TargetMesh->MarkPackageDirty();
```

- The above code really changed the normals, but the rendering is not working.

- Try to investigate the skeletal mesh rendering related code
```
Engine\Internal\Streaming\SkeletalMeshUpdate.cpp
Engine\Private\Components\SkeletalMeshComponent.cpp
Engine\Private\GPUSkinCache.cpp
  void FGPUSkinCache::DoDispatch(FRHICommandList& RHICmdList)
Engine\Private\Rendering\SkeletalMeshLODImporterData.cpp
Engine\Private\SkeletalMesh.cpp
Engine\Private\SkeletalRenderGPUSkin.cpp
  void FSkeletalMeshObjectGPUSkin::ProcessUpdatedDynamicData
Engine\Private\SkeletalRenderCPUSkin.cpp
Engine\Public\SkeletalRenderPublic.h
  FSkeletalMeshObject
  
CVarCommands:
p.ClothPhysics 0 可以关闭一些没有用的调用

UWorld::InitializeActorsForPlay(const FURL & InURL, bool bResetTime, FRegisterComponentContext * Context)
  UWorld::UpdateWorldComponents(bool bRerunConstructionScripts, bool bCurrentLevelOnly, FRegisterComponentContext * Context)
   ULevel::IncrementalUpdateComponents(int NumComponentsToUpdate, bool bRerunConstructionScripts, FRegisterComponentContext * Context) 
     	ULevel::IncrementalRegisterComponents(bool bPreRegisterComponents, int NumComponentsToUpdate, FRegisterComponentContext * Context)
 	      AActor::IncrementalRegisterComponents(int NumComponentsToRegister, FRegisterComponentContext * Context)
 	        UActorComponent::RegisterComponentWithWorld(UWorld * InWorld, FRegisterComponentContext * Context)
            UActorComponent::ExecuteRegisterEvents(FRegisterComponentContext * Context)
 	            USkinnedMeshComponent::CreateRenderState_Concurrent(FRegisterComponentContext * Context) 

void USkinnedMeshComponent::CreateRenderState_Concurrent(FRegisterComponentContext* Context)


/** vertex data for rendering a single LOD */
struct FSkeletalMeshObjectLOD

/** Default GPU skinning vertex factories and matrices */
FVertexFactoryData GPUSkinVertexFactories;
  		/** one vertex factory for each chunk */
		TArray<TUniquePtr<FGPUBaseSkinVertexFactory>> VertexFactories;  // size == RenderSections' num

		/** one passthrough vertex factory for each chunk */
		TArray<TUniquePtr<FGPUSkinPassthroughVertexFactory>> PassthroughVertexFactories;

void FSkeletalMeshObjectGPUSkin::FVertexFactoryData::InitVertexFactories(
	const FVertexFactoryBuffers& VertexBuffers,
	const TArray<FSkelMeshRenderSection>& Sections,
	ERHIFeatureLevel::Type InFeatureLevel,
	FGPUSkinPassthroughVertexFactory::EVertexAttributeFlags VertexAttributeMask,
	ESkeletalMeshGPUSkinTechnique GPUSkinTechnique)

VertexBuffers: contains TargetMesh->GetResourceForRendering()->LODRenderData[LODIndex]
VertexAttributeMask: contains Position | Tangent flag


- ENQUEUE_RENDER_COMMAND(InitGPUSkinVertexFactory)
void FSkeletalMeshObjectGPUSkin::CreateVertexFactory(
	TArray<TUniquePtr<FGPUBaseSkinVertexFactory>>& VertexFactories,
	TArray<TUniquePtr<FGPUSkinPassthroughVertexFactory>>* PassthroughVertexFactories,
	const FSkeletalMeshObjectGPUSkin::FVertexFactoryBuffers& VertexBuffers,
	ERHIFeatureLevel::Type FeatureLevel,
	FGPUSkinPassthroughVertexFactory::EVertexAttributeFlags VertexAttributeMask,
	uint32 BaseVertexIndex,
	bool bUsedForPassthroughVertexFactory)
{
	FGPUBaseSkinVertexFactory* VertexFactory = nullptr;
	GPUSkinBoneInfluenceType BoneInfluenceType = VertexBuffers.SkinWeightVertexBuffer->GetBoneInfluenceType();
	if (BoneInfluenceType == GPUSkinBoneInfluenceType::DefaultBoneInfluence)
	{
		VertexFactory = new TGPUSkinVertexFactory<GPUSkinBoneInfluenceType::DefaultBoneInfluence>(FeatureLevel, VertexBuffers.NumVertices, BaseVertexIndex, bUsedForPassthroughVertexFactory);
	}
	else
	{
		VertexFactory = new TGPUSkinVertexFactory<GPUSkinBoneInfluenceType::UnlimitedBoneInfluence>(FeatureLevel, VertexBuffers.NumVertices, BaseVertexIndex, bUsedForPassthroughVertexFactory);
	}
	VertexFactories.Add(TUniquePtr<FGPUBaseSkinVertexFactory>(VertexFactory));

	// Allocate optional passthrough vertex factory, if PassthroughVertexFactories is non-null
	FGPUSkinPassthroughVertexFactory* NewPassthroughVertexFactory = AllocatePassthroughVertexFactory(PassthroughVertexFactories, FeatureLevel, VertexAttributeMask);

	// Setup the update data for enqueue
	FDynamicUpdateVertexFactoryData VertexUpdateData(VertexFactory, VertexBuffers);

	// update vertex factory components and sync it
	ENQUEUE_RENDER_COMMAND(InitGPUSkinVertexFactory)(UE::RenderCommandPipe::SkeletalMesh,
		[VertexUpdateData, NewPassthroughVertexFactory](FRHICommandList& RHICmdList)
		{
			FGPUSkinDataType Data;
			InitGPUSkinVertexFactoryComponents(&Data, VertexUpdateData.VertexBuffers, VertexUpdateData.VertexFactory);
			VertexUpdateData.VertexFactory->SetData(RHICmdList, &Data);
			VertexUpdateData.VertexFactory->InitResource(RHICmdList);

			InitPassthroughVertexFactory_RenderThread(NewPassthroughVertexFactory, VertexUpdateData.VertexFactory, RHICmdList);
		}
	);
}







```

FVector4f TangentZ = VertexBuffers.StaticVertexBuffers->StaticMeshVertexBuffer.VertexTangentZ(1093);


- 看看Serialize是否有问题
```

	FSkeletalMeshLODRenderData::Serialize(FArchive& Ar, UObject* Owner, int32 Idx)
		FSkeletalMeshLODRenderData::SerializeStreamedData

```



