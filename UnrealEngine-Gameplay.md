学习一下ActionRoguelike-UE5.3游戏代码

## C++

- USaveGame和USSaveGameSubsystem
```
在USSaveGameSubsystem::LoadSaveGame(FString InSlotName /*= ""*/)中先检查是否存在，
  如果存在, 使用如下方式获取SaveGame:
    		CurrentSaveGame = Cast<USSaveGame>(UGameplayStatics::LoadGameFromSlot(CurrentSlotName, 0));
  否则创建一个USSaveGame:
        CurrentSaveGame = CastChecked<USSaveGame>(UGameplayStatics::CreateSaveGameObject(USSaveGame::StaticClass()));
  然后遍历所有Actor:
    

```


- 关于Bone
```
void USkeletalMeshComponent::ResetToRefPose()
{
	if (USkeletalMesh* SkelMesh = GetSkeletalMeshAsset())
	{
PRAGMA_DISABLE_DEPRECATION_WARNINGS
		BoneSpaceTransforms = SkelMesh->GetRefSkeleton().GetRefBonePose();
		//Mini RefreshBoneTransforms (the bit we actually care about)
		SkelMesh->FillComponentSpaceTransforms(BoneSpaceTransforms, FillComponentSpaceTransformsRequiredBones, GetEditableComponentSpaceTransforms());
PRAGMA_ENABLE_DEPRECATION_WARNINGS
		bNeedToFlipSpaceBaseBuffers = true; // Have updated space bases so need to flip
		FlipEditableSpaceBases();
	}
}



void USkinnedAsset::FillComponentSpaceTransforms(const TArray<FTransform>& InBoneSpaceTransforms,
												 const TArray<FBoneIndexType>& InFillComponentSpaceTransformsRequiredBones, 
												 TArray<FTransform>& OutComponentSpaceTransforms) const


void USkinnedMeshComponent::FlipEditableSpaceBases()
{
	if (bNeedToFlipSpaceBaseBuffers)
	{
		bNeedToFlipSpaceBaseBuffers = false;

		if (bDoubleBufferedComponentSpaceTransforms)
		{
			CurrentReadComponentTransforms = CurrentEditableComponentTransforms;
			CurrentEditableComponentTransforms = 1 - CurrentEditableComponentTransforms;

			// copy to other buffer if we dont already have a valid set of transforms
			if (!bHasValidBoneTransform)
			{
				GetEditableComponentSpaceTransforms() = GetComponentSpaceTransforms();
				GetEditableBoneVisibilityStates() = GetBoneVisibilityStates();
				bBoneVisibilityDirty = false;
			}
			// If we have changed bone visibility, then we need to reflect that next frame
			else if(bBoneVisibilityDirty)
			{
				GetEditableBoneVisibilityStates() = GetBoneVisibilityStates();
				bBoneVisibilityDirty = false;
			}
		}
		else
		{
			// save previous transform if it's valid
			if (bHasValidBoneTransform)
			{
				PreviousComponentSpaceTransformsArray = GetComponentSpaceTransforms();
				PreviousBoneVisibilityStates = GetBoneVisibilityStates();
			}

			CurrentReadComponentTransforms = CurrentEditableComponentTransforms = 0;

			// if we don't have a valid transform, we copy after we write, so that it doesn't cause motion blur
			if (!bHasValidBoneTransform)
			{
				PreviousComponentSpaceTransformsArray = GetComponentSpaceTransforms();
				PreviousBoneVisibilityStates = GetBoneVisibilityStates();
			}
		}

		BoneTransformUpdateMethodQueue.Add(EBoneTransformUpdateMethod::AnimationUpdate);
		// Bone revision number needs to be updated immediately, because dynamic updates on components are run in parallel later,
		// for a follower component it relies on its lead component to be up-to-date, so updating the lead component revision number here guarantees it.
		UpdateBoneTransformRevisionNumber();
	}
}

void USkeletalMeshComponent::RecalcRequiredBones(int32 LODIndex)
{
	if (!GetSkeletalMeshAsset())
	{
		return;
	}

	ComputeRequiredBones(RequiredBones, FillComponentSpaceTransformsRequiredBones, LODIndex, /*bIgnorePhysicsAsset=*/ false);

	// Reset our animated pose to the reference pose
	PRAGMA_DISABLE_DEPRECATION_WARNINGS
	BoneSpaceTransforms = GetSkeletalMeshAsset()->GetRefSkeleton().GetRefBonePose();
	PRAGMA_ENABLE_DEPRECATION_WARNINGS

	// Make sure no other parallel task is ongoing since we need to reset the shared required bones
	// and they might be in use
	HandleExistingParallelEvaluationTask(true, false);

	// If we had cached our shared bone container, reset it
	if (SharedRequiredBones)
	{
		SharedRequiredBones->Reset();
	}

	// make sure animation requiredBone to mark as dirty
	if (AnimScriptInstance)
	{
		AnimScriptInstance->RecalcRequiredBones();
	}

	for (UAnimInstance* LinkedInstance : LinkedInstances)
	{
		LinkedInstance->RecalcRequiredBones();
	}

	if (PostProcessAnimInstance)
	{
		PostProcessAnimInstance->RecalcRequiredBones();
	}

	// when RecalcRequiredBones happened
	// this should always happen
	MarkRequiredCurveUpToDate();
	bRequiredBonesUpToDate = true;

	// Invalidate cached bones.
	ClearCachedAnimProperties();

	OnLODRequiredBonesUpdate.Broadcast(this, LODIndex, RequiredBones);
}




void USkeletalMeshComponent::ComputeRequiredBones(TArray<FBoneIndexType>& OutRequiredBones, TArray<FBoneIndexType>& OutFillComponentSpaceTransformsRequiredBones, int32 LODIndex, bool bIgnorePhysicsAsset) const
{
	OutRequiredBones.Reset();
	OutFillComponentSpaceTransformsRequiredBones.Reset();

  USkeletalMesh* SkelMesh = GetSkeletalMeshAsset();
  FSkeletalMeshRenderData* SkelMeshRenderData = GetSkeletalMeshRenderData();
  // The list of bones we want is taken from the predicted LOD level.
  FSkeletalMeshLODRenderData& LODData = SkelMeshRenderData->LODRenderData[LODIndex];
  OutRequiredBones = LODData.RequiredBones;
  
  // Add virtual bones
  GetRequiredVirtualBones(SkelMesh, OutRequiredBones);

  const UPhysicsAsset* const PhysicsAsset = GetPhysicsAsset();
	// If we have a PhysicsAsset, we also need to make sure that all the bones used by it are always updated, as its used
	// by line checks etc. We might also want to kick in the physics, which means having valid bone transforms.
	if (!bIgnorePhysicsAsset && PhysicsAsset)
	{
		GetPhysicsRequiredBones(SkelMesh, PhysicsAsset, OutRequiredBones);
	}


	// Purge invisible bones and their children
	// this has to be done before mirror table check/physics body checks
	// mirror table/phys body ones has to be calculated
	ExcludeHiddenBones(this, SkelMesh, OutRequiredBones);

	// Get socket bones set to animate and bones required to fill the component space base transforms
	TArray<FBoneIndexType> NeededBonesForFillComponentSpaceTransforms;
	GetSocketRequiredBones(SkelMesh, OutRequiredBones, NeededBonesForFillComponentSpaceTransforms);

	// Gather any bones referenced by shadow shapes
	GetShadowShapeRequiredBones(this, OutRequiredBones);

	// Ensure that we have a complete hierarchy down to those bones.
	FAnimationRuntime::EnsureParentsPresent(OutRequiredBones, SkelMesh->GetRefSkeleton());




}


struct FReferenceSkeleton
{
	FReferenceSkeleton(bool bInOnlyOneRootAllowed = true)
		:bOnlyOneRootAllowed(bInOnlyOneRootAllowed)
	{}

private:
	//RAW BONES: Bones that exist in the original asset
	/** Reference bone related info to be serialized **/
	TArray<FMeshBoneInfo>	RawRefBoneInfo;
	/** Reference bone transform **/
	TArray<FTransform>		RawRefBonePose;

	//FINAL BONES: Bones for this skeleton including user added virtual bones
	/** Reference bone related info to be serialized **/
	TArray<FMeshBoneInfo>	FinalRefBoneInfo;
	/** Reference bone transform **/
	TArray<FTransform>		FinalRefBonePose;

	/** TMap to look up bone index from bone name. */
	TMap<FName, int32>		RawNameToIndexMap;
	TMap<FName, int32>		FinalNameToIndexMap;

	// cached data to allow virtual bones to be built into poses
	TArray<FBoneIndexType>  RequiredVirtualBones;
	TArray<FVirtualBoneRefData> UsedVirtualBoneData;

  ...
}
```








- USInteractionComponent
```
这个对象是USCharacter的子Component:
	InteractionComp = CreateDefaultSubobject<USInteractionComponent>("InteractionComp");


有一个UFunction的tag第一次看到，是BlueprintNativeEvent:
	UFUNCTION(BlueprintCallable, BlueprintNativeEvent)
	void Interact(APawn* InstigatorPawn);
在C++中可以调用函数ExecuteInteract


这里也是另一些Tag:
	// Reliable - Will always arrive, eventually. Request will be re-sent unless an acknowledgment was received.
	// Unreliable - Not guaranteed, packet can get lost and won't retry.
	UFUNCTION(Server, Reliable)
	void ServerInteract(AActor* InFocus);

C++文件中对应一个这样的代码：对应按F会触发调用这个函数。
void USInteractionComponent::ServerInteract_Implementation(AActor* InFocus)
{
	if (InFocus == nullptr)
	{
		GEngine->AddOnScreenDebugMessage(-1, 1.0f, FColor::Red, "No Focus Actor to interact.");
		return;
	}

	APawn* MyPawn = CastChecked<APawn>(GetOwner());
	ISGameplayInterface::Execute_Interact(InFocus, MyPawn); // Execute_Interact貌似是UE自动生成的函数？？
}



类型转化：
  APawn* MyPawn = CastChecked<APawn>(GetOwner());


获取最新的相机位置：
构造函数中
	// Since we use Camera info in Tick we want the most up to date camera position for tracing
	PrimaryComponentTick.TickGroup = TG_PostUpdateWork;


找到可交互的Actor并显示Widget:

void USInteractionComponent::FindBestInteractable()
{
	const bool bDebugDraw = CVarDebugDrawInteraction.GetValueOnGameThread();

	FCollisionObjectQueryParams ObjectQueryParams;
	ObjectQueryParams.AddObjectTypesToQuery(CollisionChannel);

	FVector EyeLocation;
	FRotator EyeRotation;
	GetOwner()->GetActorEyesViewPoint(EyeLocation, EyeRotation);

	FVector End = EyeLocation + (EyeRotation.Vector() * TraceDistance);

	TArray<FHitResult> Hits;

	FCollisionShape Shape;
	Shape.SetSphere(TraceRadius);

	bool bBlockingHit = GetWorld()->SweepMultiByObjectType(Hits, EyeLocation, End, FQuat::Identity, ObjectQueryParams, Shape);

	FColor LineColor = bBlockingHit ? FColor::Green : FColor::Red;

	// Clear ref before trying to fill
	FocusedActor = nullptr;

	for (FHitResult Hit : Hits)
	{
		if (bDebugDraw)
		{
			DrawDebugSphere(GetWorld(), Hit.ImpactPoint, TraceRadius, 32, LineColor, false, 0.0f);
		}

		AActor* HitActor = Hit.GetActor();
		if (HitActor)
		{
			if (HitActor->Implements<USGameplayInterface>())
			{
				FocusedActor = HitActor;
				break;
			}
		}
	}

	if (FocusedActor)
	{
		if (DefaultWidgetInstance == nullptr && ensure(DefaultWidgetClass))
		{
			DefaultWidgetInstance = CreateWidget<USWorldUserWidget>(GetWorld(), DefaultWidgetClass);
		}

		if (DefaultWidgetInstance)
		{
			DefaultWidgetInstance->AttachedActor = FocusedActor;

			if (!DefaultWidgetInstance->IsInViewport())
			{
				DefaultWidgetInstance->AddToViewport();
			}
		}
	}
	else
	{
		if (DefaultWidgetInstance)
		{
			DefaultWidgetInstance->RemoveFromParent();
		}
	}


	if (bDebugDraw)
	{
		DrawDebugLine(GetWorld(), EyeLocation, End, LineColor, false, 2.0f, 0, 0.0f);
	}
}

奇怪的调用路线：
  Execute_Interact -> AActor::ProcessEvent -> UObject::ProcessEvent -> UFunction::Invoke --> SGameplayInterface::execInteract --> ASPowerup_HealthPotion::Interact_Implementation

```



- UGameViewportClient
```
/**
 * A game viewport (FViewport) is a high-level abstract interface for the
 * platform specific rendering, audio, and input subsystems.
 * GameViewportClient is the engine's interface to a game viewport.
 * Exactly one GameViewportClient is created for each instance of the game.  The
 * only case (so far) where you might have a single instance of Engine, but
 * multiple instances of the game (and thus multiple GameViewportClients) is when
 * you have more than one PIE window running.
 *
 * Responsibilities:
 * propagating input events to the global interactions list
 *
 * @see UGameViewportClient
 */

在这个项目中派生了一个类USGameViewportClient, 重载了 virtual void Tick(float DeltaTime) override;
void USGameViewportClient::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

	if (USignificanceManager* SignificanceManager = USignificanceManager::Get(World))
	{
		// Update once per frame, using only Player 0
		if (APlayerController* PC = UGameplayStatics::GetPlayerController(World, 0))
		{
			FVector ViewLocation;
			FRotator ViewRotation;
			PC->GetPlayerViewPoint(ViewLocation, ViewRotation);

			// Viewpoints
			TArray<FTransform> TransformArray;
			TransformArray.Emplace(ViewRotation, ViewLocation, FVector::OneVector);

			SignificanceManager->Update(TArrayView<FTransform>(TransformArray));
		}
	}
}

```

- CVars
```
写法1：

/* Allows us to force significance on all classes to quickly compare the performance differences as if the system was disabled */
static float GForcedSignificance = -1;
static FAutoConsoleVariableRef CVarSignificanceManager_ForceSignificance(
	TEXT("SigMan.ForceSignificance"),
	GForcedSignificance,
	TEXT("Force significance on all managed objects. -1 is default, 0-4 is hidden, lowest, medium, highest.\n"),
	ECVF_Cheat
	);



写法2：
static TAutoConsoleVariable<float> CVarDamageMultiplier(TEXT("game.DamageMultiplier"), 1.0f, TEXT("Global Damage Modifier for Attribute Component."), ECVF_Cheat);
const float DamageMultiplier = CVarDamageMultiplier.GetValueOnGameThread();

static TAutoConsoleVariable CVarActorPoolingEnabled(
	TEXT("game.ActorPooling"),
	true,
	TEXT("Enable actor pooling for selected objects."),
	ECVF_Default);
CVarActorPoolingEnabled.GetValueOnAnyThread() && WorldContextObject->GetWorld()->IsNetMode(NM_Standalone);
```

